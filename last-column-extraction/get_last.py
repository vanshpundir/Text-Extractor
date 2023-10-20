import cv2
import numpy as np
import pytesseract

# Specify the Tesseract executable path (replace '/usr/bin/tesseract' with your actual Tesseract path if it's different)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Read the input image
image_path = '/home/dell/Documents/github/Text-Extractor/image_processing/rotated.jpg'
img = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to the grayscale image
_, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Perform morphological operations to clean up the image
kernel = np.ones((4, 1), np.uint8)
processed_image = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

# Find contours in the processed image
contours, _ = cv2.findContours(processed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Initialize variables to store last column bounds
last_column_x_start = 0
last_column_x_end = 0

# Find the rightmost contours based on their x-coordinates
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    # Set a threshold based on the width of the table to identify the last column
    threshold_width = 200 # Adjust this value based on your table's characteristics
    if w > threshold_width and x > last_column_x_end:
        last_column_x_start = x
        last_column_x_end = x + w

# Crop the last column based on the determined boundaries
last_column_crop = img[:, last_column_x_start:last_column_x_end]

# Save the cropped last column image
cv2.imwrite('last_column.png', last_column_crop)
