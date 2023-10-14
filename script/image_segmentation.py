import cv2
import numpy as np

# set image path
path = "/Users/vansh/PycharmProjects/Text-Extractor/image_processing/"
filename = "8_image_without_lines_noise_removed.jpg"

# read input image
inputimage = cv2.imread(path+filename)

# deep copy for results:
inputimagecopy = inputimage.copy()

# convert bgr to grayscale:
grayscaleimage = cv2.cvtcolor(inputimage, cv2.color_bgr2gray)

# threshold via otsu:
threshvalue, binaryimage = cv2.threshold(grayscaleimage, 0, 255, cv2.thresh_binary_inv+cv2.thresh_otsu)