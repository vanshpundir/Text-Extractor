import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('/Users/vansh/PycharmProjects/Text-Extractor/images/extract_data/IMG_9518.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Reduce noise in the image
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

# Detect edges in the image
edged = cv2.Canny(bfilter, 30, 200)

# Find the contours of the detected edges
contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the bounding rectangle of the largest contour
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)

# Crop the image to the bounding rectangle of the largest contour
cropped_image = cv2.getRectSubPix(img, (w, h), (x + w // 2, y + h // 2))

# Display the cropped image
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.show()
