import cv2
import numpy as np

# Load the image
image = cv2.imread('/Users/vansh/PycharmProjects/Text-Extractor/images/extract_data/cropped_image.jpg', 0)  # Load as grayscale

# Apply Gaussian blur to the image to reduce noise


# Use adaptive thresholding to binarize the image
_, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Create a horizontal kernel for morphological operations to remove horizontal lines
horizontal_kernel = np.ones((1, 50), np.uint8)
opened_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

# Create a vertical kernel for morphological operations to remove vertical lines
vertical_kernel = np.ones((50, 1), np.uint8)
opened_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

# Subtract the opened images from the original to get lines as white
result = cv2.subtract(binary, opened_horizontal)
result = cv2.subtract(result, opened_vertical)

# Display the result using imshow
denoised_image = cv2.medianBlur(result, 3)

cv2.imshow("demsofdn", denoised_image)
cv2.waitKey()
cv2.destroyAllWindows()