import cv2
import numpy as np
import math


def logistic_cdf(x, mu, sigma):
    return 1 / (1 + math.exp(-(x - mu) / sigma))


def dist_rgb(c1, c2):
    distB = math.pow(c1[0].astype(np.float), 2) - math.pow(c2[0].astype(np.float), 2)
    distG = math.pow(c1[1].astype(np.float), 2) - math.pow(c2[1].astype(np.float), 2)
    distR = math.pow(c1[2].astype(np.float), 2) - math.pow(c2[2].astype(np.float), 2)
    dist = math.pow(math.pow(distB, 2) + math.pow(distG, 2) + math.pow(distR, 2), 1 / 4)
    return dist


def minDistColor(img, mask, referenceColor):
    # print(mask.shape)
    # print(img.shape)
    mask = mask.reshape((1, mask.shape[0], mask.shape[1]))
    arr = img[tuple(mask == 1)]
    diff = np.power(arr.astype(np.float), 2) - np.power(referenceColor.astype(np.float), 2)
    minArg = np.argmin(np.power(np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2) + np.power(diff[:, 2], 2), 1 / 4))
    return arr[minArg]


def hasSimilarBgr(A, B):
    """
    Returns true if one element of the list of BGR colors A is contained in a list of BGR colors B
    :param A: list of BGR colors
    :param B: list of BGR colors
    :return: true if one element of the list of BGR colors A is contained in a list of BGR colors B, false otherwise
    """
    match = np.array([np.in1d(A[:, 0], B[:, 0]),
                      np.in1d(A[:, 1], B[:, 1]),
                      np.in1d(A[:, 2], B[:, 2])])
    return np.any(np.logical_and(np.logical_and(match[0, :], match[1, :]), match[2, :]))


def on_color_tolerance(tl):
    global tolerance
    tolerance = tl


def on_softness(sl):
    global softness_level
    if (sl == 0):
        softness_level = 1
    else:
        softness_level = sl


def on_color_cast_level(ccl):
    global color_cast_level
    color_cast_level = ccl
    print(color_cast_level)


def select_background_color(action, x, y, flags, userdata):
    global frame
    global selected_hue
    global selected_saturation
    global selected_value

    if action == cv2.EVENT_LBUTTONDOWN:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_frame)
        selected_hue = np.mean(h[y - 10:y + 10, x - 10:x + 10])
        selected_saturation = np.mean(s[y - 10:y + 10, x - 10:x + 10])
        selected_value = np.mean(v[y - 10:y + 10, x - 10:x + 10])


def generateTrimap(action, x, y, flags, userdata):
    global frame
    global trimap
    global tolerance
    global softness_level
    if action == cv2.EVENT_LBUTTONDOWN:
        trimap = np.ones(frame.shape[0:2]) * 127

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        backgroundHSV = hsv_frame[y, x, :]
        print(frame[y, x, :])
        iH, iS, iV = cv2.split(hsv_frame)
        hb = np.where((iH <= backgroundHSV[0] + tolerance) & (iH >= backgroundHSV[0] - tolerance), 1, 0)
        print(hb.shape)
        sb = np.where(iS >= 150, 1, 0)
        print(sb.shape)
        backgroundMask = np.logical_and(hb, sb).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        backgroundMask = cv2.morphologyEx(backgroundMask, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        backgroundMask = cv2.morphologyEx(backgroundMask, cv2.MORPH_OPEN, kernel, iterations=4)

        foregroundMask = np.where(backgroundMask == 1, 0, 1).astype(np.uint8)
        foregroundMask = cv2.morphologyEx(foregroundMask, cv2.MORPH_OPEN, kernel, iterations=4)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        foregroundMask = cv2.erode(foregroundMask, kernel, iterations=8)
        backgroundMask = cv2.erode(backgroundMask, kernel, iterations=8)
        trimap = np.where(backgroundMask, 0, trimap)
        trimap = np.where(foregroundMask, 255, trimap)
        trimap = trimap.astype(np.uint8)

        alphaMask = np.zeros(trimap.shape)

        for y in range(0, trimap.shape[0]):
            for x in range(0, trimap.shape[1]):
                if trimap[y, x] == 255:
                    alphaMask[y, x] = 255
                elif trimap[y, x] == 127:
                    starty = 0 if y - 60 < 0 else y - 60
                    endy = trimap.shape[0] - 1 if y + 60 >= trimap.shape[0] else y + 60
                    startx = 0 if x - 60 < 0 else x - 60
                    endx = trimap.shape[1] - 1 if x + 60 >= trimap.shape[1] else x + 60
                    localForegroundMask = foregroundMask[starty:endy, startx:endx]
                    localForegroundMask = localForegroundMask.reshape(
                        (1, localForegroundMask.shape[0], localForegroundMask.shape[1]))
                    meanForeground = np.mean(
                        frame[starty:endy, startx:endx][tuple(localForegroundMask == 1)].reshape(-1, 3), axis=0).astype(
                        np.uint8)
                    localBackgroundMask = backgroundMask[starty:endy, startx:endx]
                    localBackgroundMask = localBackgroundMask.reshape(
                        (1, localBackgroundMask.shape[0], localBackgroundMask.shape[1]))
                    meanBackground = np.mean(
                        frame[starty:endy, startx:endx][tuple(localBackgroundMask == 1)].reshape(-1, 3), axis=0).astype(
                        np.uint8)

                    # Try minimum euclidean distance between the two arrays (fg and bg)
                    # foregroundColors = frame[starty:endy, startx:endx][tuple(localForegroundMask == 1)].reshape(-1, 3)
                    backgroundColors = frame[starty:endy, startx:endx][tuple(localBackgroundMask == 1)].reshape(-1, 3)
                    if hasSimilarBgr(frame[y, x].reshape(1, 3), backgroundColors):
                        print("continue")
                        alphaMask[y, x] = np.uint8(0)
                        continue

                    distMean = dist_rgb(meanForeground, meanBackground)

                    if distMean == 0:
                        distMean = 0.1

                    distX = dist_rgb(frame[y, x], meanBackground) / distMean

                    # 0.16 is the solution of solve(1/(1+exp(-(0-0.5)/s))<0.05)
                    sigma = 0.16 * softness_level / 100
                    alpha = logistic_cdf(distX, 0.5, sigma)
                    if 0.99 > alpha > 0.05:
                        # foregroundNeighborhood = highlyLikelyForeground[starty:endy, startx:endx]
                        neighborhood = frame[starty:endy, startx:endx]
                        frame[y, x] = minDistColor(neighborhood, foregroundMask[starty:endy, startx:endx], frame[y, x])

                    alpha = alpha * 255
                    if alpha < 0:
                        alpha = 0
                    elif alpha > 255:
                        alpha = 255
                    alpha = round(alpha)
                    alpha = np.uint8(alpha)
                    print(alpha)

                    alphaMask[y, x] = alpha

        # highlyLikelyAlphaMask = np.where(alphaMask == 255, 1, 0)
        alphaMask = alphaMask.astype(np.uint8)
        alphaMask3d = cv2.merge((alphaMask, alphaMask, alphaMask)).astype(np.float)
        print(alphaMask)
        cv2.namedWindow("trimap")
        cv2.imshow("trimap", trimap)
        cv2.namedWindow("result")
        white = (np.ones(frame.shape) * 255).astype(np.uint8)

        # pivot points for X-Coordinates
        originalValue = np.array([0, 50, 100, 150, 200, 255])

        fullRange = np.arange(0, 256)
        color_cast_percentage = color_cast_level / 100
        gCurve = np.array([0,
                           50 - color_cast_percentage * 30,
                           100 - color_cast_percentage * 50,
                           150 - color_cast_percentage * 40,
                           200 - color_cast_percentage * 20,
                           255])
        gLUT = np.interp(fullRange, originalValue, gCurve)
        gChannel = frame[:, :, 1]
        gChannel = cv2.LUT(gChannel, gLUT)
        frame[:, :, 1] = gChannel

        show_background = cv2.add(cv2.multiply(white.astype(np.float), (1 - alphaMask3d / 255)),
                                  cv2.multiply(frame.astype(np.float), alphaMask3d / 255))
        bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alphaMask

        cv2.imshow("result", show_background.astype(np.uint8))
        cv2.waitKey(0)


cap = cv2.VideoCapture('greenscreen-demo.mp4')
cv2.namedWindow("Chroma_keying")
# highgui function called when mouse events occur
cv2.setMouseCallback("Chroma_keying", generateTrimap)
cv2.createTrackbar("color_tolerance", "Chroma_keying", 0, 100, on_color_tolerance)
cv2.createTrackbar("softness_level", "Chroma_keying", 0, 100, on_softness)
cv2.createTrackbar("color_cast_level", "Chroma_keying", 0, 100, on_color_cast_level)
k = 0
# loop until escape character is pressed
ret, frame = cap.read()
preview = frame.copy()
selected_hue = 0
selected_value = 0
selected_saturation = 0
tolerance = 0
softness_level = 1
color_cast_level = 0
trimap = np.zeros(frame.shape[0:2])

while k != 27:
    cv2.imshow("Chroma_keying", preview)

    k = cv2.waitKey(20) & 0xFF
    # print(k)

cv2.destroyAllWindows()
cap.release()
