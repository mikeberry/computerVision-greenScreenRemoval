import cv2
import numpy as np


def render_preview():
    global frame
    global preview
    global selected_hue
    global selected_saturation
    global selected_value
    global tolerance_level
    # First create the image with alpha channel
    rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

    # Then assign the mask to the last channel of the image
    rgba[:, :, 3] = alpha_data
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hue = max(selected_hue - tolerance_level / 100 * 30, 0)
    upper_hue = min(selected_hue + tolerance_level / 100 * 30, 255)
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
    global tolerance_level
    tolerance_level = tl
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
    if action == cv2.EVENT_LBUTTONDOWN:
        trimap = np.ones(frame.shape[0:2])*127
        ib, ig, ir = cv2.split(frame)
        backgroundBGR = frame[y, x, :]
        print(backgroundBGR)
        tolerance = 0
        bb = np.where((ib <= backgroundBGR[0] + tolerance) & (ib >= backgroundBGR[0] - tolerance), 1, 0)
        bg = np.where((ig <= backgroundBGR[1] + tolerance) & (ig >= backgroundBGR[1] - tolerance), 1, 0)
        br = np.where((ir <= backgroundBGR[2] + tolerance) & (ir >= backgroundBGR[2] - tolerance), 1, 0)
        backgroundMask = np.logical_or(np.logical_or(bb, bg), br).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        backgroundMask = cv2.morphologyEx(backgroundMask, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        backgroundMask = cv2.morphologyEx(backgroundMask, cv2.MORPH_OPEN, kernel, iterations=4)
        # backgroundMask = cv2.morphologyEx(backgroundMask, cv2.MORPH_OPEN, kernel)
        # backgroundMask = cv2.morphologyEx(backgroundMask, cv2.MORPH_OPEN, kernel)
        # backgroundMask = cv2.morphologyEx(backgroundMask, cv2.MORPH_OPEN, kernel)
        foregroundMask = np.where(backgroundMask == 1, 0, 1).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        foregroundMask = cv2.erode(foregroundMask, kernel, iterations=3)
        trimap = np.where(backgroundMask,0,trimap)
        trimap = np.where(foregroundMask,255,trimap)
        trimap = trimap.astype(np.uint8)

        alphaMask = np.zeros(trimap.shape)

        meanBackground = [19, 255, 69]
        print(foregroundMask.shape)
        print(frame.shape)
        maskedForeground = cv2.multiply(frame, cv2.merge((foregroundMask, foregroundMask, foregroundMask)))
        # print(maskedForeground)
        sumOfFgpix = np.sum(foregroundMask)
        mfB, mfG, mfR = cv2.split(maskedForeground)
        meanForeground = [np.sum(mfB) / sumOfFgpix, np.sum(mfG) / sumOfFgpix, np.sum(mfR) / sumOfFgpix]
        meanForeground = np.asarray(meanForeground).astype(np.uint8)
        print(meanForeground)

        for y in range(0,trimap.shape[0]):
            for x in range(0,trimap.shape[1]):
                if trimap[y,x] == 255:
                    alphaMask[y,x] = 255
                elif trimap[y,x] == 127:
                    print("yay 127")
                    imageColor = frame[y,x]
                    alpha = np.dot((imageColor - meanBackground),(meanForeground-meanBackground))/(np.linalg.norm(meanForeground-meanBackground)**2)*255

                    if alpha > 255:
                        alpha = 255
                    if alpha < 0:
                        alpha = 0
                    alpha = round(alpha)
                    print(alpha)
                    alphaMask[y,x] = alpha
        alphaMask = alphaMask.astype(np.uint8)
        alphaMask3d = cv2.merge((alphaMask,alphaMask,alphaMask)).astype(np.float)
        print(alphaMask)
        cv2.namedWindow("trimap")
        cv2.imshow("trimap", alphaMask)
        cv2.namedWindow("result")
        white = (np.ones(frame.shape)*255).astype(np.uint8)
        show_background = cv2.add(cv2.multiply(white.astype(np.float),(1-alphaMask3d/255)),cv2.multiply(frame.astype(np.float),alphaMask3d/255))
        bgra = cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)
        bgra[:,:,3] = alphaMask

        cv2.imshow("result",show_background.astype(np.uint8))
        cv2.waitKey(0)


cap = cv2.VideoCapture('greenscreen-asteroid.mp4')
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
tolerance_level = 0
trimap = np.zeros(frame.shape[0:2])

while k != 27:
    cv2.imshow("Chroma_keying", preview)

    k = cv2.waitKey(20) & 0xFF
    # print(k)

cv2.destroyAllWindows()
cap.release()
