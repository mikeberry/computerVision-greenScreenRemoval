import cv2
import numpy as np


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

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        backgroundHSV = hsv_frame[y, x, :]
        print(backgroundHSV)
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
        # calculating object histogram
        print("backgroundMask:")
        print((foregroundMask * 255).astype(np.uint8).dtype)
        print(backgroundMask.dtype)
        # backgroundMask2d = cv2.merge((backgroundMask, backgroundMask))
        roihist = cv2.calcHist(images=[hsv_frame], channels=[0, 1], mask=(backgroundMask * 255).astype(np.uint8), histSize=[180, 256], ranges=[0, 180, 0, 256])
        print(roihist.shape)
        hHist,_ = np.histogram(np.random.normal(54, 10, 1000),bins=256)
        print(hHist)
        sHist,_ = np.histogram(np.random.normal(176, 10, 1000),bins=256)
        print(sHist)
        print(hHist.shape)
        roihist[0] = hHist
        roihist[1] = sHist
        print(np.max(iH))
        print(np.min(iH))
        print(np.max(iS))
        print(np.min(iS))
        # roihist = cv2.calcHist(images=[maskedBackground], channels=[0, 1], mask=None, histSize=[180, 256], ranges=[0, 180, 0, 256])
        print(roihist.dtype)
        print(roihist.shape)
        print(roihist[0])
        print(roihist[1])
        cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
        print(roihist[0])
        print(roihist[1])
        alphaMask = cv2.calcBackProject([hsv_frame], [0, 1], roihist, [0, 180, 0, 256], 1)
        alphaMask3d = cv2.merge((alphaMask, alphaMask, alphaMask)).astype(np.float)
        print(alphaMask)
        cv2.namedWindow("trimap")
        cv2.imshow("trimap", maskedForeground)
        cv2.namedWindow("result")
        white = (np.ones(frame.shape) * 255).astype(np.uint8)
        show_background = cv2.add(cv2.multiply(white.astype(np.float), (alphaMask3d / 255)),
                                  cv2.multiply(frame.astype(np.float), 1 - alphaMask3d / 255))
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
