import cv2
import numpy as np


def render_preview():
    global frame
    global preview
    global selected_hue
    global selected_saturation
    global selected_value
    global tolerance_level
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hue = max(selected_hue - tolerance_level / 100 * 30,0)
    upper_hue = min(selected_hue + tolerance_level / 100 * 30,255)
    lower_value = max(selected_value - 100,0)
    upper_value = min(selected_value + 100,255)
    lower_saturation = max(selected_value - 50,0)
    upper_saturation = min(selected_value + 50,255)
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
        selected_hue = np.mean(h[y-10:y+10, x-10:x+10])
        selected_saturation = np.mean(s[y-10:y+10, x-10:x+10])
        selected_value = np.mean(v[y-10:y+10, x-10:x+10])
        render_preview()


cap = cv2.VideoCapture('greenscreen-asteroid.mp4')
cv2.namedWindow("Chroma_keying")
# highgui function called when mouse events occur
cv2.setMouseCallback("Chroma_keying", select_background_color)
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

while k != 27:
    cv2.imshow("Chroma_keying", preview)

    k = cv2.waitKey(20) & 0xFF
    # print(k)

cv2.destroyAllWindows()
cap.release()
