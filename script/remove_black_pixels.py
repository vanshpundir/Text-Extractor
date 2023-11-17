import cv2
import numpy as np

# Read the image
image = cv2.imread('/home/shivam/Documents/Github/Text-Extractor/script/rotated_image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply GaussianBlur to reduce noise and improve line detection
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply edge detection to highlight features
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

# Use probabilistic Hough Transform to detect lines
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

# Create masks for horizontal and vertical lines
horizontal_mask = np.array([line[0][1] for line in lines], dtype=np.float32) < np.pi / 4  # Horizontal lines have angles close to 0 or π
vertical_mask = np.array([line[0][1] for line in lines], dtype=np.float32) > np.pi / 2  # Vertical lines have angles close to π/2

# Draw the detected horizontal lines on a copy of the original image
horizontal_lines_image = image.copy()
for line, is_horizontal in zip(lines, horizontal_mask):
    if is_horizontal:
        x1, y1, x2, y2 = line[0]
        cv2.line(horizontal_lines_image, (x1, y1), (x2, y2), 255, 2)

# Draw the detected vertical lines on a copy of the original image
vertical_lines_image = image.copy()
for line, is_vertical in zip(lines, vertical_mask):
    if is_vertical:
        x1, y1, x2, y2 = line[0]
        cv2.line(vertical_lines_image, (x1, y1), (x2, y2), 255, 2)

# Save the results
cv2.imwrite('output_horizontal_lines.jpg', horizontal_lines_image)
cv2.imwrite('output_vertical_lines.jpg', vertical_lines_image)
