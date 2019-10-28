import cv2
import numpy as np
import math
from progress_bar import ProgressBar


def logistic_cdf(x, mu, sigma):
    return 1 / (1 + math.exp(-(x - mu) / sigma))


def dist_rgb(c1, c2):
    dist_b = math.pow(c1[0].astype(np.float), 2) - math.pow(c2[0].astype(np.float), 2)
    dist_g = math.pow(c1[1].astype(np.float), 2) - math.pow(c2[1].astype(np.float), 2)
    dist_r = math.pow(c1[2].astype(np.float), 2) - math.pow(c2[2].astype(np.float), 2)
    dist = math.pow(math.pow(dist_b, 2) + math.pow(dist_g, 2) + math.pow(dist_r, 2), 1 / 4)
    return dist


def min_dist_color(img, mask, reference_color):
    # print(mask.shape)
    # print(img.shape)
    # mask = mask.reshape((1, mask.shape[0], mask.shape[1]))
    arr = img[tuple(mask == 1)]
    if len(arr) == 0:
        return reference_color
    diff = np.power(arr.astype(np.float), 2) - np.power(reference_color.astype(np.float), 2)
    minArg = np.argmin(np.power(np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2) + np.power(diff[:, 2], 2), 1 / 4))
    return arr[minArg]


def has_similar_bgr(A, B):
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


def generate_matted_image(original_image, selected_background_hsv, tolerance, new_background_image):
    print("generate_matted_image")
    trimap = np.ones(original_image.shape[0:2]) * 127
    hsv_frame = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    i_h, i_s, i_v = cv2.split(hsv_frame)
    hb = np.where((i_h <= selected_background_hsv[0] + tolerance) & (i_h >= selected_background_hsv[0] - tolerance), 1,
                  0)
    sb = np.where(i_s >= 120, 1, 0)
    background_mask = np.logical_and(hb, sb).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel, iterations=4)

    foreground_mask = np.where(background_mask == 1, 0, 1).astype(np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel, iterations=4)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground_mask = cv2.erode(foreground_mask, kernel, iterations=8)
    background_mask = cv2.erode(background_mask, kernel, iterations=12)
    trimap = np.where(background_mask, 0, trimap)
    trimap = np.where(foreground_mask, 255, trimap)
    trimap = trimap.astype(np.uint8)
    # TODO add  pixels with the most common background colors to background
    background_colors, counts, = np.unique(frame[tuple(background_mask.reshape(1,background_mask.shape[0], background_mask.shape[1]) == 1)], axis=0, return_counts=True)
    background_colors = background_colors[tuple(counts.reshape(1,-1)>10000)]
    counts = counts[tuple(counts.reshape(1,-1)>10000)]
    print(background_colors)
    print(counts)
    print(background_colors.shape)
    print(counts.shape)
    match = np.array([np.in1d(frame[:,:, 0], background_colors[:, 0]),
                      np.in1d(frame[:,:, 1], background_colors[:, 1]),
                      np.in1d(frame[:,:, 2], background_colors[:, 2])])
    mask = np.logical_and(np.logical_and(match[0, :], match[1, :]), match[2, :])
    mask = mask.reshape((frame.shape[0],frame.shape[1]))
    background_mask = np.logical_or(background_mask, mask)
    trimap = np.where(mask, 0, trimap)

    alpha_mask = trimap.copy()

    ys, xs = np.where(trimap == 127)
    progress_bar = ProgressBar(len(ys), 20)
    progress = 0
    h = trimap.shape[0]
    w = trimap.shape[1]
    for i in range(0,len(ys)):
        x = xs[i]
        y = ys[i]
        progress_bar.update_progress_bar(progress)
        progress = progress + 1
        start_y = 0 if y - 60 < 0 else y - 60
        end_y = h if y + 60 >= h else y + 60
        start_x = 0 if x - 60 < 0 else x - 60
        end_x = w if x + 60 >= w else x + 60
        local_w = end_x - start_x
        local_h = end_y - start_y
        neighborhood = original_image[start_y:end_y, start_x:end_x]
        local_foreground_mask = foreground_mask[start_y:end_y, start_x:end_x]
        local_foreground_mask = local_foreground_mask.reshape((1, local_h, local_w))
        mean_foreground = np.mean(
            neighborhood[tuple(local_foreground_mask == 1)].reshape(-1, 3),
            axis=0).astype(np.uint8)
        local_background_mask = background_mask[start_y:end_y, start_x:end_x]
        local_background_mask = local_background_mask.reshape((1, local_h, local_w))
        mean_background = np.mean(
            neighborhood[tuple(local_background_mask == 1)].reshape(-1, 3),
            axis=0).astype(np.uint8)

        # Try minimum euclidean distance between the two arrays (fg and bg)
        # foregroundColors = frame[start_y:end_y, start_x:end_x][tuple(local_foreground_mask == 1)].reshape(-1, 3)
        local_background_colors = neighborhood[tuple(local_background_mask == 1)].reshape(-1,3)
        if has_similar_bgr(original_image[y, x].reshape(1, 3), local_background_colors):
            # print("continue")
            alpha_mask[y, x] = np.uint8(0)
            continue

        dist_mean = dist_rgb(mean_foreground, mean_background)

        if dist_mean == 0:
            dist_mean = 0.1

        dist_x = dist_rgb(original_image[y, x], mean_background) / dist_mean

        # 0.16 is the solution of solve(1/(1+exp(-(0-0.5)/s))<0.05)
        sigma = 0.16 * softness_level / 100
        alpha = logistic_cdf(dist_x, 0.5, sigma)
        if 0.99 > alpha > 0.05:
            # foregroundNeighborhood = highlyLikelyForeground[start_y:end_y, start_x:end_x]

            original_image[y, x] = min_dist_color(neighborhood, local_foreground_mask,
                                                  original_image[y, x])

        alpha = alpha * 255
        if alpha < 0:
            alpha = 0
        elif alpha > 255:
            alpha = 255
        alpha = round(alpha)
        alpha = np.uint8(alpha)
        # print(alpha)

        alpha_mask[y, x] = alpha

    # highlyLikelyAlphaMask = np.where(alpha_mask == 255, 1, 0)
    alpha_mask = alpha_mask.astype(np.uint8)
    alpha_mask3d = cv2.merge((alpha_mask, alpha_mask, alpha_mask)).astype(np.float)
    # print(alpha_mask)
    cv2.namedWindow("trimap")
    cv2.imshow("trimap", trimap)
    cv2.namedWindow("result")

    # pivot points for X-Coordinates
    original_value = np.array([0, 50, 100, 150, 200, 255])

    full_range = np.arange(0, 256)
    color_cast_percentage = color_cast_level / 100
    g_curve = np.array([0,
                        50 - color_cast_percentage * 30,
                        100 - color_cast_percentage * 50,
                        150 - color_cast_percentage * 40,
                        200 - color_cast_percentage * 20,
                        255])
    g_lut = np.interp(full_range, original_value, g_curve)
    g_channel = original_image[:, :, 1]
    g_channel = cv2.LUT(g_channel, g_lut)
    original_image[:, :, 1] = g_channel

    result = cv2.add(cv2.multiply(new_background_image.astype(np.float), (1 - alpha_mask3d / 255)),
                     cv2.multiply(original_image.astype(np.float), alpha_mask3d / 255))
    bgra = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha_mask
    return result


def convert_video(video, background_image, out_path):
    global selected_hsv
    global tolerance

    fps = math.ceil(video.get(cv2.CAP_PROP_FPS))
    print(fps)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))
    i = 0
    while (True):
        print("frame: " + str(i))
        ret, frame = video.read()
        if ret:
            matted_image = generate_matted_image(frame, selected_hsv,
                                                 tolerance, background_image)
            out.write(np.uint8(matted_image))
        else:
            print("finished")
            break
        i += 1


def on_color_tolerance(tl):
    global tolerance
    tolerance = tl


def on_softness(sl):
    global softness_level
    if sl == 0:
        softness_level = 1
    else:
        softness_level = sl


def on_color_cast_level(ccl):
    global color_cast_level
    color_cast_level = ccl


def select_background_color(action, x, y, flags, userdata):
    global frame
    global selected_hsv
    global background_image
    global tolerance

    if action == cv2.EVENT_LBUTTONDOWN:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_frame)
        selected_hsv[0] = np.mean(h[y - 10:y + 10, x - 10:x + 10])
        selected_hsv[1] = np.mean(s[y - 10:y + 10, x - 10:x + 10])
        selected_hsv[2] = np.mean(v[y - 10:y + 10, x - 10:x + 10])
        matted_image = generate_matted_image(frame, selected_hsv,
                                             tolerance, background_image)
        cv2.imshow("result", matted_image.astype(np.uint8))
        cv2.waitKey(0)


cap = cv2.VideoCapture('greenscreen-demo.mp4')
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
selected_hsv = np.array([0, 0, 0])
tolerance = 0
softness_level = 1
color_cast_level = 0
try:
    background_image = cv2.imread("background.jpg")
    background_image = cv2.resize(background_image, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
except:
    print("could not find background.jpg. Assuming white background")
    background_image = white = (np.ones(frame.shape) * 255).astype(np.uint8)

while k != 27:
    cv2.imshow("Chroma_keying", preview)

    k = cv2.waitKey(20) & 0xFF
    # print(k)
    if k == 116:
        convert_video(cap, background_image, out_path="matted.avi")

cv2.destroyAllWindows()
cap.release()
