import cv2
import pytesseract

# Read the image
img = cv2.imread("/Users/vansh/PycharmProjects/Text-Extractor/images/extract_data/test.png")

# Use pytesseract to extract orientation
text = pytesseract.image_to_osd(img)
lines = text.splitlines()

# Find the line that contains orientation information
for line in lines:
    if "Orientation in degrees" in line:
        rotation_angle = float(line.split(":")[1].strip())
        break

# Get center and dimensions
height, width = img.shape[:2]
center = (width / 2, height / 2)

# Rotate image
rot_mat = cv2.getRotationMatrix2D(center, -rotation_angle, 1)
rotated = cv2.warpAffine(img, rot_mat, (width, height))

# Save rotated image
cv2.imwrite('rotated.jpg', rotated)
