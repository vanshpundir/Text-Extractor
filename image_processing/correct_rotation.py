import cv2
import numpy as np
import math# Readim image
img = cv2.imread("/Users/vansh/PycharmProjects/Text-Extractor/images/extract_data/IMG_9518.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find edges using Canny edge detection
edges = cv2.Canny(gray, 50, 200)

# Run Hough transform on edges to find lines
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=5)

# Extract angles and lengths of lines
angles = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = math.atan2(y2 - y1, x2 - x1)
    length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    angles.append(angle)

# Find average angle
angle_sum = sum(angles)
num_angles = len(angles)
avg_angle = angle_sum / num_angles

# Calculate rotation angle needed to make image horizontal
rotation_angle = -np.rad2deg(avg_angle)+1
print(rotation_angle)
# Get center and dimensions
height, width = img.shape[:2]
center = (width/2, height/2)

# Rotate image
rot_mat = cv2.getRotationMatrix2D(center, rotation_angle, 1)
rotated = cv2.warpAffine(img, rot_mat, (width, height))

# Save rotated image
cv2.imwrite('rotated.jpg', rotated)
