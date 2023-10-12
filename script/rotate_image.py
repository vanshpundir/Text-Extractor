import cv2

# Read the image
img = cv2.imread("/Users/vansh/PycharmProjects/Text-Extractor/images/extract_data/image_text.png")

# Get the image dimensions
height, width, channels = img.shape

# Calculate the center of the image
center = (width // 2, height // 2)

# Get the rotation matrix
rot_mat = cv2.getRotationMatrix2D(center, -1, 1)

# Rotate the image
rotated_img = cv2.warpAffine(img, rot_mat, (width, height))

# Save the rotated image
cv2.imwrite("rotated_image.jpg", rotated_img)
