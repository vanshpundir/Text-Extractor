import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
# Upload the image in Google Colab


image_path  = '/Users/vansh/PycharmProjects/Text-Extractor/images/handwritten_img/Screenshot 2023-10-14 at 10.48.19 AM.png'

# Load the image
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to binarize the image
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11, 2.0)
# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create an empty canvas to draw the contours on
contour_image = np.zeros_like(image)

# List to store extracted digits
extracted_digits = []

con = 0
# Loop through the contours
for i, contour in enumerate(contours):
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Extract the digit from the original image using the bounding box
    digit = image[y:y+h, x:x+w]

    # Save the digit image with a unique filename (e.g., "digit0.png", "digit1.png", etc.)
    filename = f"/Users/vansh/PycharmProjects/Text-Extractor/script/segmented_images/digit{i}.png"
    cv2.imwrite(filename, digit)

    digit_gray = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
    cv2.imshow("img",digit_gray)

    pil_image = Image.fromarray(digit_gray)
    # Convert grayscale to RGB
    digit_rgb = pil_image.convert('RGB')
    print(digit_gray.shape)
    # Append the extracted digit to the list
    extracted_digits.append(digit)

    # Draw the contour on the canvas
    cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)

# Display the image with contours using Matplotlib
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Now, you have a list of extracted digits in the 'extracted_digits' variable for further processing or recognition.
