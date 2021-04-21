import cv2
import pytesseract
import numpy as np
from scipy.ndimage import interpolation as inter
from skimage.restoration import denoise_tv_chambolle
from imutils import perspective
from imutils import contours
import imutils
from sklearn.cluster import k_means

# pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = '.apt/usr/bin/tesseract'


def show_image(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)


def correct_skew(image, delta=3, limit=45):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    return best_angle, rotated


def threshold_image(image, n1, n2):
    threshold1 = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, n1, n2
    )
    threshold2 = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, n1, -n2
    )
    edges1 = cv2.Canny(threshold1, 30, 200)
    edge_count1 = np.count_nonzero(edges1)
    edges2 = cv2.Canny(threshold2, 30, 200)
    edge_count2 = np.count_nonzero(edges2)
    if edge_count1 <= edge_count2:
        return threshold1
    return threshold2


# Takes  uint8  ,gray
# returns uint8 , gray
def rescaling(image):
    resized = cv2.resize(image, (700, 700), interpolation=cv2.INTER_CUBIC)
    return resized


def medianFiltering(img_noisy1):
    # Obtain the number of rows and columns
    # of the image
    m, n = img_noisy1.shape

    # Traverse the image. For every 3X3 area,
    # find the median of the pixels and
    # replace the ceter pixel by the median
    img_new1 = np.zeros([m, n])
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            temp = [
                img_noisy1[i - 1, j - 1],
                img_noisy1[i - 1, j],
                img_noisy1[i - 1, j + 1],
                img_noisy1[i, j - 1],
                img_noisy1[i, j],
                img_noisy1[i, j + 1],
                img_noisy1[i + 1, j - 1],
                img_noisy1[i + 1, j],
                img_noisy1[i + 1, j + 1],
            ]

            temp = sorted(temp)
            img_new1[i, j] = temp[4]

    return img_new1.astype(np.uint8)


# takes float64 , gray
# returns uint8  , gray
def normalize(img):
    #    info = np.iinfo(data.dtype) # Get the information of the incoming image type
    #    data = data.astype(np.float64) / info.max # normalize the data to 0 - 1
    img = 255 * img  # Now scale by 255
    img = img.astype(np.uint8)
    return img


# takes uint8  gray
# returns flaot64  gray
def de_noise(img):
    denoised_image_tv = denoise_tv_chambolle(img, weight=0.1, multichannel=True)
    # denoised_image_tv = denoised_image_tv.astype(np.uint8)
    return denoised_image_tv


# Contrasting and Filtering
# takes uint8  gray
# returns uint8  gray
def clah(img):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    return clahe.apply(img)


def find_table(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = threshold_image(gray, 9, 5)
    # edged = cv2.Canny(gray, 50, 100)

    output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    max_w, max_m, max_h = 0, 0, 0
    for i in range(1, numLabels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
            text = "examining component {}/{} (background)".format(i + 1, numLabels)
        # otherwise, we are examining an actual connected component
        else:
            text = "examining component {}/{}".format(i + 1, numLabels)
        # print a status message update for the current connected
        # component
        # print("[INFO] {}".format(text))
        # extract the connected component statistics and centroid for
        # the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        m = w + h
        if max_m < m:
            (X, Y, max_w, max_h, I, max_m) = x, y, w, h, i, m

    output = img.copy()
    cv2.rectangle(output, (X, Y), (X + max_w, Y + max_h), (0, 255, 0), 3)
    componentMask = (labels == I).astype("uint8") * 255
    cnts = cv2.findContours(
        componentMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue
        # compute the rotated bounding box of the contour
        orig = img.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
            # (orig, "Image")

    # contours,_= cv2.findContours(componentMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # mask = np.zeros(componentMask.shape, np.uint8)
    # cv2.fillPoly(mask, contours, 255)
    # kernel = np.ones((15, 15), np.uint8)
    # erosion = cv2.erode(mask, kernel, iterations=1)
    # n=cv2.bitwise_and(erosion,thresh)

    # show our output image and connected component mask
    # (output, "Output")
    # (thresh, "Thresh")
    # (componentMask, "Connected Component")
    return (X, Y, max_w, max_h)


def new_cut_table(img):
    (x, y, width, height) = find_table(img)
    circles = [[x, y], [x + width, y], [x, y + height], [x + width, y + height]]
    rate = float(700 / height)
    width = int(rate * width)
    if width > 1300:
        width = 1300
    height = 700
    pts1 = np.float32([circles[0], circles[1], circles[2], circles[3]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    output = cv2.warpPerspective(img, matrix, (width, height))
    return output


def Remove_impurities(img):
    output = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    h_key, w_key = (img.shape[0] / 6), (img.shape[1] / 4)
    mask = np.zeros(img.shape, dtype="uint8")
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if h > h_key or w > w_key:
            mask = cv2.bitwise_or(mask, (labels == i).astype("uint8") * 255)

    mask = cv2.bitwise_not(mask)
    mask = cv2.bitwise_and(mask, img)
    return mask


# Morph open to remove noise
# takes uint8 gray
def morphOpen(image):
    # kernel = np.ones((2,2),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.dilate(image, kernel, iterations=1)
    # erosion = cv2.erode(dilation,kernel,iterations = 1)
    return closing


circles = np.zeros((4, 2), np.int)
counter = 0


def mousePointers(event, x, y, flags, parameters):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        circles[counter] = x, y
        counter = counter + 1


def cut_table(image):
    scale_percent = 120  # percent of original size
    # width = int((circles[1][0]-circles[0][0]) * scale_percent / 100)
    # height = int((circles[2][1]-circles[0][1])  * scale_percent / 100)

    width, height = int(circles[1][0] - circles[0][0]), int(
        circles[2][1] - circles[0][1]
    )
    rate = float(700 / height)
    width = int(rate * width)
    if width > 1300:
        width = 1300
    height = 700
    pts1 = np.float32([circles[0], circles[1], circles[2], circles[3]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    output = cv2.warpPerspective(image, matrix, (width, height))
    return output


def remove_lines(image):
    vertical = image
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    vertical = cv2.erode(vertical, verticalStructure, iterations=1)
    vertical = cv2.dilate(vertical, verticalStructure, iterations=1)
    verticalStructure = np.ones((9, 9), np.uint8)
    vertical = cv2.dilate(vertical, verticalStructure, iterations=1)
    vertical = cv2.bitwise_not(vertical)

    horizontal = image
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    horizontal = cv2.erode(horizontal, verticalStructure, iterations=1)
    horizontal = cv2.dilate(horizontal, verticalStructure, iterations=1)
    horizontalStructure = np.ones((8, 8), np.uint8)
    horizontal = cv2.dilate(horizontal, horizontalStructure, iterations=1)
    horizontal = cv2.bitwise_not(horizontal)

    thresh = cv2.bitwise_and(image, image, mask=vertical)
    thresh = cv2.bitwise_and(thresh, thresh, mask=horizontal)
    thresh = cv2.bitwise_not(thresh)
    return thresh


def histogramequlization(img):

    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)

    hsv_image = cv2.merge([h, s, v])
    hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    img = cv2.fastNlMeansDenoisingColored(hsv_image, None, 10, 10, 9, 21)

    return img


def tesseract(thresh):
    custom_config = r"--oem 3 --psm 6"
    string = pytesseract.image_to_string(thresh, config=custom_config)
    return string


def preprocess(img):
    # read image
    img = rescaling(img)
    # show_image(img)
    
    img = histogramequlization(img)
    # show_image(img, " equalizeHist")
    
    img = new_cut_table(img)
    # show_image(img, "cut")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # show_image(img, " gray")
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    img = de_noise(img)
    # show_image(img, "Denoise")

    img = normalize(img)
    # show_image(img, "Normalized")

    img = threshold_image(img, 35, 11)
    # show_image(img, "threshold_image")

    img = Remove_impurities(img)
    # show_image(img, "threshold_image")

    img = remove_lines(img)
    # show_image(img, "remove_lines")

    img = morphOpen(img)
    # show_image(img, "morph")

    text = tesseract(img)
    return text