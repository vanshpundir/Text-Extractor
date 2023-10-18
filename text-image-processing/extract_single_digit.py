import cv2
import os
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Load the image
image = cv2.imread('/Users/vansh/PycharmProjects/Text-Extractor/text-image-processing/processed_image/single_digit/left_image10.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Find contours with a minimum contour area threshold
contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize variables to keep track of the largest contour and its size
largest_contour = None
largest_contour_area = 0

# Set a minimum height and width threshold
min_height = 10
min_width = 10

# Iterate through all the contours
for contour in contours:
    # Calculate the bounding box for the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate the area of the bounding box (considering both height and width)
    bounding_box_area = w * h

    # Check if this bounding box area is larger than the current largest
    if bounding_box_area > largest_contour_area and h >= min_height and w >= min_width:
        largest_contour_area = bounding_box_area
        largest_contour = contour

# Check if the largest contour exists
if largest_contour is not None:
    # Create a mask for the largest contour
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Extract the region within the contour
    result_image = cv2.bitwise_and(image, mask)

    # Define the output directory
    output_directory = '/Users/vansh/PycharmProjects/Text-Extractor/text-image-processing/single_digit_processed'

    # Check if the output directory exists; if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save the image with the largest contour to the output directory
    output_path = os.path.join(output_directory, 'largest_contour_image.png')
    cv2.imwrite(output_path, result_image)

    # Log a success message
    logger.info(f'Largest contour image saved to: {output_path}')
else:
    # Log a warning for empty or too small images
    logger.warning('Empty or too small image detected.')
