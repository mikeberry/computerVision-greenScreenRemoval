import cv2
import numpy as np
import math

def gaussian(x, mu, sigma):
    return 1/(sigma * math.sqrt(2*math.pi))*math.exp(-1/2*math.pow((x-mu)/sigma,2))

def logistic_cdf(x, mu, sigma):
    return 1/(1+math.exp(-(x-mu)/sigma))

def render_preview():
    global frame
    global preview
    global selected_hue
    global selected_saturation
    global selected_value
    global tolerance
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hue = max(selected_hue - tolerance / 100 * 30, 0)
    upper_hue = min(selected_hue + tolerance / 100 * 30, 255)
    lower_value = max(selected_value - 100, 0)
    upper_value = min(selected_value + 100, 255)
    lower_saturation = max(selected_value - 50, 0)
    upper_saturation = min(selected_value + 50, 255)
    lower_key = np.array([lower_hue, lower_saturation, lower_value])
    upper_key = np.array([upper_hue, upper_saturation, upper_value])
    mask = cv2.inRange(hsv_frame, lower_key, upper_key)
    preview = frame.copy()
    preview[mask != 0] = np.array([0, 0, 0])


def on_color_tolerance(tl):
    global tolerance
    tolerance = tl
    render_preview()


def on_softness(softness_level):
    print(softness_level)


def on_color_cast_level(color_cast_level):
    print(color_cast_level)


def select_background_color(action, x, y, flags, userdata):
    global frame
    global selected_hue
    global selected_saturation
    global selected_value

    if action == cv2.EVENT_LBUTTONDOWN:
        print("Hello")
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_frame)
        selected_hue = np.mean(h[y - 10:y + 10, x - 10:x + 10])
        selected_saturation = np.mean(s[y - 10:y + 10, x - 10:x + 10])
        selected_value = np.mean(v[y - 10:y + 10, x - 10:x + 10])
        render_preview()


def generateTrimap(action, x, y, flags, userdata):
    global frame
    global trimap
    global tolerance
    if action == cv2.EVENT_LBUTTONDOWN:
        trimap = np.ones(frame.shape[0:2]) * 127
        # ib, ig, ir = cv2.split(frame)
        # backgroundBGR = frame[y, x, :]
        # print(backgroundBGR)
        # bb = np.where((ib <= backgroundBGR[0] + tolerance) & (ib >= backgroundBGR[0] - tolerance), 1, 0)
        # bg = np.where((ig <= backgroundBGR[1] + tolerance) & (ig >= backgroundBGR[1] - tolerance), 1, 0)
        # br = np.where((ir <= backgroundBGR[2] + tolerance) & (ir >= backgroundBGR[2] - tolerance), 1, 0)
        # backgroundMask = np.logical_or(np.logical_or(bb, bg), br).astype(np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # backgroundMask = cv2.morphologyEx(backgroundMask, cv2.MORPH_CLOSE, kernel)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        # backgroundMask = cv2.morphologyEx(backgroundMask, cv2.MORPH_OPEN, kernel, iterations=4)
        # foregroundMask = np.where(backgroundMask == 1, 0, 1).astype(np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        # foregroundMask = cv2.erode(foregroundMask, kernel, iterations=3)
        # trimap = np.where(backgroundMask, 0, trimap)
        # trimap = np.where(foregroundMask, 255, trimap)
        # trimap = trimap.astype(np.uint8)

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        backgroundHSV = hsv_frame[y, x, :]
        print(frame[y, x, :])
        iH, iS, iV = cv2.split(hsv_frame)
        hb = np.where((iH<= backgroundHSV[0]+tolerance) & (iH>= backgroundHSV[0]- tolerance),1,0)
        print(hb.shape)
        sb = np.where(iS >= 150,1,0)
        print(sb.shape)
        backgroundMask = np.logical_and(hb,sb).astype(np.uint8)
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

        maskedForeground = cv2.multiply(frame, cv2.merge((foregroundMask, foregroundMask, foregroundMask)))
        maskedBackground = cv2.multiply(frame, cv2.merge((backgroundMask, backgroundMask, backgroundMask)))

        for y in range(0, trimap.shape[0]):
            for x in range(0, trimap.shape[1]):
                if trimap[y, x] == 255:
                    alphaMask[y, x] = 255
                elif trimap[y, x] == 127:
                    # meanBackground = [19, 255, 69]
                    # print(foregroundMask.shape)
                    # print(frame.shape)
                    # print(maskedForeground)
                    starty = 0 if y - 60 < 0 else y -60
                    endy = trimap.shape[0]-1 if y + 60 >= trimap.shape[0] else y + 60
                    startx = 0 if x -60 <0 else x -60
                    endx = trimap.shape[1]-1 if x + 60 >= trimap.shape[1] else x + 60
                    localMaskedForeground = maskedForeground[starty:endy,startx:endx]
                    sumOfFgpix = np.sum(foregroundMask[starty:endy,startx:endx])
                    mfB, mfG, mfR = cv2.split(localMaskedForeground)
                    meanForeground = [np.sum(mfB) / sumOfFgpix, np.sum(mfG) / sumOfFgpix, np.sum(mfR) / sumOfFgpix]
                    meanForeground = np.asarray(meanForeground).astype(np.uint8)

                    localMaskedBackground = maskedBackground[starty:endy,startx:endx]
                    sumOfBgpix = np.sum(backgroundMask[starty:endy, startx:endx])
                    mbB, mbG, mbR = cv2.split(localMaskedBackground)
                    meanBackground = [np.sum(mbB) / sumOfBgpix, np.sum(mbG) / sumOfBgpix, np.sum(mbR) / sumOfBgpix]
                    meanBackground = np.asarray(meanBackground).astype(np.uint8)

                    # print(meanForeground)
                    # print(meanBackground)

                    imageColorLin = frame[y, x].astype(np.float)/(255)
                    meanBackgroundLin = meanBackground.astype(np.float)/(255)
                    meanForegroundLin = meanForeground.astype(np.float)/(255)
                    # alpha = np.dot((np.abs(imageColorLin - meanBackgroundLin)), (np.abs(meanForegroundLin - meanBackgroundLin))).astype(float) / (
                    #         np.linalg.norm(meanForegroundLin - meanBackgroundLin).astype(float) ** 2)
                    # gHist, _ = np.histogram(np.random.normal(meanForeground[1], 20, 1000), bins=256, density=True)
                    # print(gHist)
                    # alpha = 1 - sum(gHist[0:frame[y, x][1]])
                    meanGForeBack = (meanBackground[1] + meanForeground[1])/2
                    # alpha = 1-(gaussian(frame[y, x][1],meanBackground[1],5)/gaussian(meanBackground[1],meanBackground[1],5))
                    #alpha = 1-(gaussian(frame[y, x][1],meanGForeBack,20)/gaussian(meanGForeBack,meanGForeBack,20))
                    #Try the distance:
                    distMean = abs(meanForeground[1].astype(int)- meanBackground[1].astype(int))
                    distX = abs(frame[y, x][1].astype(int) - meanBackground[1].astype(int))

                    #gHist, _ = np.histogram(np.random.normal(distMean, 20, 1000), bins=256, density=True)
                    #print(gHist)
                    #alpha = sum(gHist[0:math.floor(distX)])

                    alpha = logistic_cdf(distX, distMean/2,5)

                    if alpha > 0.95:
                        alpha = 1.0
                    elif alpha < 0.05:
                        alpha = 0.0
                    alpha = alpha * 255
                    if alpha > 255:
                        alpha = 255
                    if alpha < 0:
                        alpha = 0
                    alpha = round(alpha)
                    alpha = np.uint8(alpha)
                    print(alpha)
                    #print(meanBackground)

                    alphaMask[y, x] = alpha
        alphaMask = alphaMask.astype(np.uint8)
        alphaMask3d = cv2.merge((alphaMask, alphaMask, alphaMask)).astype(np.float)
        print(alphaMask)
        cv2.namedWindow("trimap")
        cv2.imshow("trimap", trimap)
        cv2.namedWindow("result")
        white = (np.ones(frame.shape) * 255).astype(np.uint8)
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
trimap = np.zeros(frame.shape[0:2])

while k != 27:
    cv2.imshow("Chroma_keying", preview)

    k = cv2.waitKey(20) & 0xFF
    # print(k)

cv2.destroyAllWindows()
cap.release()
