import cv2
import pytesseract
import numpy as np
from skimage.restoration import denoise_tv_chambolle
import imutils


print(cv2.__version__)


# pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"


def show_image(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)

def Remove_impurities(img):
    output = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    h_key,w_key=(img.shape[0]/6),(img.shape[1]/6)
    mask=np.zeros(img.shape, dtype="uint8")
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if  h > h_key or w >w_key :
         mask = cv2.bitwise_or(mask,(labels == i).astype("uint8") * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask=cv2.bitwise_not(mask)

    return mask



def correct_skew(image,gray,gray1,impurities):
    Kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    Dilate = cv2.dilate(image, Kernel, iterations=3)
    Contours, hierarchy = cv2.findContours(Dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    Contours = sorted(Contours, key=cv2.contourArea, reverse=True)
    largestContour = Contours[0]
    angle = cv2.minAreaRect(largestContour)[-1]
    print(angle)
    if angle > 45:
        angle =  angle-90
    else:
        angle = angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, .98)
    rotated_without = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated_gray = cv2.warpAffine(gray, M, (w , h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated_gray1 = cv2.warpAffine(gray1, M, (w, h),
                                  flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated_impurities = cv2.warpAffine(impurities, M, (w, h),
                                  flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated_impurities = np.uint8(rotated_impurities)
    ret, rotated_impurities = cv2.threshold(rotated_impurities, 127, 255, cv2.THRESH_BINARY)

    return rotated_gray,rotated_gray1,rotated_impurities,rotated_without


def threshold_image(image,n1,n2):
    threshold1 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, n1, n2)
    #a=Remove_impurities(dilate)
    #show_image(a, " a")
    threshold2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, n1, -n2)
    edges1 = cv2.Canny(threshold1, 30, 200)
    edge_count1 = np.count_nonzero(edges1)
    edges2 = cv2.Canny(threshold2, 30, 200)
    edge_count2 = np.count_nonzero(edges2)
    if edge_count1 <= edge_count2:
        return threshold1
    return threshold2

def tesseract(thresh):
    custom_config  = r'--oem 3 -l eng --psm 6'
    string = pytesseract.image_to_string(thresh, config=custom_config)
    return string
def sort_contours(cnts, method):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)
def segment_line(gray,gray1,img_without,img,w_image,h_image):

    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    sobely = np.uint8(sobely)
    ret, sobely = cv2.threshold(sobely, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    opening = cv2.morphologyEx(sobely, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    dil = cv2.dilate(opening, kernel, iterations=1)
    output = cv2.connectedComponentsWithStats(dil, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    w_key =img.shape[1] / 6
    mask = np.zeros(img.shape, dtype="uint8")
    h_key =0
    string=""
    cropped2=[]
    thresh2=[]
    final=[]
    n=0
    for i in range(1, numLabels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        if  w > w_key:
            y = stats[i, cv2.CC_STAT_TOP]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            mask = cv2.bitwise_or(mask, (labels == i).astype("uint8") * 255)
            cropped = gray1[h_key:y+h, 0: w_image]
            cropped_without = img_without[h_key:y + h, 0: w_image]
            croped_imp=img[h_key:y + h, 0: w_image]
            h_key=y
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 1))
            morph1 = cv2.dilate(cropped_without, kernel, iterations=10)
            morph1=cv2.bitwise_not(morph1)
            morph1=cv2.dilate(morph1, kernel, iterations=10)
            morph1 = cv2.bitwise_not(morph1)
            morph1 = cv2.dilate(morph1, kernel, iterations=20)
            cntrs1 = cv2.findContours(morph1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntrs1 = imutils.grab_contours(cntrs1)
            cntrs1 = sorted(cntrs1, key=cv2.contourArea, reverse=True)[:5]
            if len(cntrs1) > 0:
              (cntrs1, boundingBoxes) = sort_contours(cntrs1, "top-to-bottom")

            for c in cntrs1:
                box = cv2.boundingRect(c)
                xe, ye, we, he = box
                if he > 10:

                    ythresh = ye
                    if ye < 5:
                        ge = 0
                    else:
                        ge = 5
                    if ye + he > cropped_without.shape[0] - 5:
                        te = 0
                    else:
                        te = 5

                    cropped1 = cropped_without[ye - ge:ye + he + te, xe: xe + we]
                    cr1 = cropped[ye - ge:ye + he + te, xe: xe + we]
                    cr_imp=croped_imp[ye - ge:ye + he + te, xe: xe + we]
                    if he< 70:
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
                        morph2 = cv2.dilate(cropped1, kernel, iterations=1)
                        cntrs2 = cv2.findContours(morph2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cntrs2 = imutils.grab_contours(cntrs2)
                        cntrs2 = sorted(cntrs2, key=cv2.contourArea, reverse=True)[:5]
                        (cntrs2, boundingBoxes) = sort_contours(cntrs2, "left-to-right")
                        for c2 in cntrs2:
                            box = cv2.boundingRect(c2)
                            xo, yo, wo, ho = box
                            if ho > 15:

                                if yo < 5:
                                    go = 0
                                else:
                                    go = 5
                                if yo + ho > cropped1.shape[0] - 5:
                                    to = 0
                                else:
                                    to = 5

                                cropped2.append(cr1[yo - go:yo + ho + to, xo: xo + wo])
                                final.append(cr_imp[yo - go:yo + ho + to, xo: xo + wo])
                                # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
                                # equ2 = clahe.apply(cropped2)

                                croped_h = cropped2[n].shape[0]
                                croped_w = cropped2[n].shape[1]
                                rate = float(100 / croped_h)
                                croped_h = 100
                                croped_w = int(rate * croped_w)
                                if croped_w > 1300:
                                    croped_w = 1300
                                cropped2[n] = cv2.resize(cropped2[n], (croped_w, croped_h),
                                                         interpolation=cv2.INTER_CUBIC)
                                final[n] = cv2.resize(final[n], (croped_w, croped_h), interpolation=cv2.INTER_CUBIC)
                                cropped2[n] = cv2.GaussianBlur(cropped2[n], (5,5),5)
                                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
                                cropped2[n]  = clahe.apply(cropped2[n] )
                                thresh2.append(
                                    cv2.threshold(cropped2[n], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
                                final[n] = cv2.bitwise_and(thresh2[n], final[n])
                                string = string + tesseract(final[n])
                                n = n + 1
                        string = string + '\n'
                    else:
                        string = string + tesseract(cropped1)


    cropped = gray[h_key:h_image, 0: w_image]

    cropped_without = img_without[h_key:h_image, 0: w_image]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 1))
    morph1 = cv2.dilate(cropped_without, kernel, iterations=10)
    morph1 = cv2.bitwise_not(morph1)
    morph1 = cv2.dilate(morph1, kernel, iterations=10)
    morph1 = cv2.bitwise_not(morph1)
    morph1 = cv2.dilate(morph1, kernel, iterations=20)
    cntrs1 = cv2.findContours(morph1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs1 = imutils.grab_contours(cntrs1)
    cntrs1 = sorted(cntrs1, key=cv2.contourArea, reverse=True)[:5]
    if len(cntrs1) >0:
      (cntrs1, boundingBoxes) = sort_contours(cntrs1, "top-to-bottom")

    for c in cntrs1:
        box = cv2.boundingRect(c)
        xe, ye, we, he = box
        if he > 15:

            ythresh = ye
            if ye < 5:
                ge = 0
            else:
                ge = 5
            if ye + he > cropped_without.shape[0] - 5:
                te = 0
            else:
                te = 5

            cropped1 = cropped_without[ye - ge:ye + he + te, xe: xe + we]
            cr1 = cropped[ye - ge:ye + he + te, xe: xe + we]
            cr_imp = croped_imp[ye - ge:ye + he + te, xe: xe + we]
            if he < 70:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
                morph2 = cv2.dilate(cropped1, kernel, iterations=1)
                cntrs2 = cv2.findContours(morph2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cntrs2 = imutils.grab_contours(cntrs2)
                cntrs2 = sorted(cntrs2, key=cv2.contourArea, reverse=True)[:5]
                (cntrs2, boundingBoxes) = sort_contours(cntrs2, "left-to-right")
                for c2 in cntrs2:
                    box = cv2.boundingRect(c2)
                    xo, yo, wo, ho = box
                    if ho > 15:

                        if yo < 5:
                            go = 0
                        else:
                            go = 5
                        if yo + ho > cropped1.shape[0] - 5:
                            to = 0
                        else:
                            to = 5

                        cropped2.append(cr1[yo - go:yo + ho + to, xo: xo + wo])
                        final.append(cr_imp[yo - go:yo + ho + to, xo: xo + wo])
                        # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
                        # equ2 = clahe.apply(cropped2)

                        croped_h = cropped2[n].shape[0]
                        croped_w = cropped2[n].shape[1]
                        rate = float(100 / croped_h)
                        croped_h = 100
                        croped_w = int(rate * croped_w)
                        if croped_w > 1300:
                            croped_w = 1300
                        cropped2[n] = cv2.resize(cropped2[n], (croped_w, croped_h),
                                                 interpolation=cv2.INTER_CUBIC)
                        final[n] = cv2.resize(final[n], (croped_w, croped_h), interpolation=cv2.INTER_CUBIC)
                        cropped2[n] = cv2.GaussianBlur(cropped2[n], (5, 5), 5)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
                        cropped2[n] = clahe.apply(cropped2[n])
                        thresh2.append(
                            cv2.threshold(cropped2[n], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])

                        cv2.destroyAllWindows()
                        final[n] = cv2.bitwise_and(thresh2[n], final[n])
                        string = string + tesseract(final[n])
                        n = n + 1
                string = string + '\n'
            else:
                string = string + tesseract(cropped1)


    return string



def rescaling(image):
    height=image.shape[0]
    width=image.shape[1]
    rate = float(680 / height)
    height = 680
    width = int(rate * width)
    if width > 1300:
        width = 1300
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    return resized,width,height



def de_noise(img):
    denoised_image_tv = denoise_tv_chambolle(img, weight=0.1, multichannel=True)
    # denoised_image_tv = denoised_image_tv.astype(np.uint8)
    return denoised_image_tv

def clah(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    return clahe.apply(img)

def morphOpen(image):
    # kernel = np.ones((2,2),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.dilate(image, kernel, iterations=1)
    # erosion = cv2.erode(dilation,kernel,iterations = 1)
    return closing

def histogramequlization(img):

    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe.apply(v)

    hsv_image = cv2.merge([h, s, v])
    img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    img = cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 9, 21)

    return img

def preprocess(image):
    img, wid, hig = rescaling(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = histogramequlization(img)
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=threshold_image(gray1,15,5)
    impurities=Remove_impurities(img)
    image_without_impurities = cv2.bitwise_and(impurities, img)
    gray1,gray,rotated_impurities,rotated_without= correct_skew(image_without_impurities,gray1,gray,impurities)
    string=segment_line(gray1,gray,rotated_without, rotated_impurities, wid, hig)
    return  string