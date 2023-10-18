import cv2
import numpy as np

def add_padding_and_display(image_path, padding_size=100):
    # Read the image using cv2
    image = cv2.imread(image_path)

    # Get the dimensions of the original image
    height, width, _ = image.shape

    # Calculate the new dimensions for the padded image
    new_width = width + 2 * padding_size
    new_height = height + 2 * padding_size

    # Create a new blank white image with the desired size
    padded_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    padded_image.fill(0)  # Fill with white color

    # Paste the original image onto the padded image with the padding
    padded_image[padding_size:padding_size+height, padding_size:padding_size+width] = image

    # Display the padded image using cv2
    cv2.imshow("Padded Image", padded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
add_padding_and_display("/Users/vansh/PycharmProjects/Text-Extractor/text-image-processing/processed_image/single_digit/left_digit_section_0.png",15)
