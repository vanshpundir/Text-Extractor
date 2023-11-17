import cv2
import os

# Define the directory path
directory_path = '/Users/vansh/PycharmProjects/Text-Extractor/text-image-processing/processed_image/horizontal_digit'

# Function to separate and save left and right digits
def process_image(image_path):
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize containers for left and right digit images
    left_digit_image = None
    right_digit_image = None

    # Separate the digits
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out small contours
        if w > 10 and h > 10:
            # Check the x-coordinate to determine if it's the left or right digit
            if x + w < image.shape[1] / 2:
                left_digit_image = image[y:y + h, x:x + w]
            elif x > image.shape[1] / 2:
                right_digit_image = image[y:y + h, x:x + w]

    # Save the left and right digit images if they exist
    if left_digit_image is not None:
        left_digit_path = os.path.join(output_directory, 'left_digit_' + os.path.basename(image_path))
        cv2.imwrite(left_digit_path, left_digit_image)

    if right_digit_image is not None:
        right_digit_path = os.path.join(output_directory, 'right_digit_' + os.path.basename(image_path))
        cv2.imwrite(right_digit_path, right_digit_image)

# Output directory for processed images
output_directory = '/Users/vansh/PycharmProjects/Text-Extractor/image_processing_text/single_digit_processed'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate over the images in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".png"):  # Process only PNG files
        image_path = os.path.join(directory_path, filename)
        process_image(image_path)

print("Processing completed.")
