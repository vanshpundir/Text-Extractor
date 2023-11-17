import cv2
import numpy as np
from entity.image import Image

class VerticalImageSplitter:
    def __init__(self, image, output_dir,image_name):
        self.image = image
        self.output_dir = output_dir
        self.image_name  = image_name

    def split_image(self):
        # Read the image
        image = self.image

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Initialize the left and right images
        left_image = np.zeros_like(gray)
        right_image = np.zeros_like(gray)

        # Iterate over the rows of the image
        for i in range(gray.shape[0]):
            # Check if the row contains at least two pixels
            if gray[i].shape[0] >= 2:
                # Find the mean of the pixels in the current row
                mean = np.mean(gray[i])
                # If the mean is greater than zero, then the row contains white pixels
                if mean > 0:
                    # Calculate the midpoint of the white pixels
                    midpoint = int(gray[i].shape[0] / 2)
                    # Split the row into left and right halves
                    left_image[i, :midpoint] = gray[i, :midpoint]
                    right_image[i, midpoint:] = gray[i, midpoint:]

        # Save the left and right images
        left_image_path = f"{self.output_dir}/left_image{self.image_name}.png"
        right_image_path = f"{self.output_dir}/right_image{self.image_name}.png"
        cv2.imwrite(left_image_path, left_image)
        cv2.imwrite(right_image_path, right_image)


if __name__ == "__main__":
    input_path = "/Users/vansh/PycharmProjects/Text-Extractor/text-image-processing/processed_image/horizontal_digit/section_4.png"
    output_directory = "image_processing_text/processed_image/single_digit"

    image_splitter = VerticalImageSplitter(input_path, output_directory)
    image_splitter.split_image()
