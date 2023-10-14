import cv2

# Load the image
image_path = '/Users/vansh/PycharmProjects/Text-Extractor/script/segmented_images/digit3.png'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to separate the digits from the background
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by their x-coordinate to get the left-to-right order
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

# Create a list to store the individual digit images
individual_digit_images = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    # Ensure the contour is large enough to be considered a digit
    if w > 10 and h > 10:
        digit_image = thresh[y:y + h, x:x + w]  # This is the inverted image
        inverted_digit_image = 255 - digit_image  # Invert black and white
        individual_digit_images.append(inverted_digit_image)

# Assuming there are two digits (8 and 3), you can separate them by their x-coordinates
if len(individual_digit_images) >= 2:
    # Sort the images by their x-coordinates
    individual_digit_images = sorted(individual_digit_images, key=lambda img: cv2.boundingRect(cv2.findNonZero(img))[0])

    # Display or save the individual digit images
    for i, digit_image in enumerate(individual_digit_images):
        cv2.imshow(f'Digit {i + 1}', digit_image)
        cv2.imwrite(f'digit_{i + 1}.png', digit_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
