import cv2
import numpy as np

# Load the image
image = cv2.imread('/Users/vansh/PycharmProjects/Text-Extractor/last_column_extraction/white_image.png')

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper range for blue color in HSV
lower_blue = np.array([40, 20, 30])
upper_blue = np.array([150, 255, 255])




# Create a mask for the blue color
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Invert the mask to get the black line
black_mask = cv2.bitwise_not(blue_mask)

# Apply the mask to the original image to extract the blue pen stroke
result = cv2.bitwise_and(image, image, mask=blue_mask)

# Convert the result to grayscale
gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)



ret, bw_img = cv2.threshold(gray_result, 0, 255, cv2.THRESH_BINARY_INV)

# converting to its binary form
bw = cv2.threshold(gray_result, 127, 255, cv2.THRESH_BINARY)
bw_img = cv2.medianBlur(bw_img,5)
cv2.imshow("Binary", bw_img)

# Apply Otsu's thresholding
# binary, _ = cv2.threshold(gray_result, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Save the resulting image
cv2.imwrite('blue_pen_stroke.jpg', bw_img)

# Display the result (optional)

cv2.waitKey(0)
cv2.destroyAllWindows()