import cv2
import numpy as np

# Load the binary image (replace the path with your image)
image = cv2.imread('/Users/vansh/PycharmProjects/Text-Extractor/last-column-extraction/try_image.png', 0)

# Apply morphological operations to remove black lines (adjust the kernel size as needed)
kernel = np.ones((6, 6), np.uint8)

# Apply dilation to reconnect the cut "3"
image = cv2.dilate(image, kernel, iterations=1)

gaussian_blur = cv2.GaussianBlur(image, (1,1),2)

sharp = cv2.addWeighted(image, 1.5,gaussian_blur, -0.5, 0)
# Save the modified image
cv2.imwrite('/Users/vansh/PycharmProjects/Text-Extractor/last-column-extraction/output_image.png', image)
