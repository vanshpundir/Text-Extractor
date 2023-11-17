import cv2
import numpy as np

# Load the image
image = cv2.imread('/Users/vansh/PycharmProjects/Text-Extractor/image_processing_text/processed_image/horizontal_digit/section_2.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize containers for left and right digit images
left_digit_image = None
right_digit_image = None

# Define the minimum width and height for contours
min_width = 5
min_height = 5

# Separate the digits
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    # Filter out small contours
    if w > min_width and h > min_height:
        # Calculate the center of the bounding box
        center_x = x + w // 2

        # Check the x-coordinate to determine if it's the left or right digit
        if center_x < image.shape[1] / 2:
            left_digit_image = image[y:y + h, x:x + w]
        else:
            right_digit_image = image[y:y + h, x:x + w]

# Check if the extracted images are too small or not found
if left_digit_image is None or right_digit_image is None:
    print("Warning: No suitable digit regions found or they are too small to process.")

# Save the left and right digit images if they exist
if left_digit_image is not None:
    cv2.imwrite('/Users/vansh/PycharmProjects/Text-Extractor/image_processing_text/single_digit_processed/left_digit.png', left_digit_image)

if right_digit_image is not None:
    cv2.imwrite('/Users/vansh/PycharmProjects/Text-Extractor/image_processing_text/single_digit_processed/right_digit.png', right_digit_image)
