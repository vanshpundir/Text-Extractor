import numpy as np
import cv2
from entity.image import Image

class Binarizer:
    def __init__(self, array):
        self.image = Image()
        self.image.array = array
        self.image = self.image.array

    def convert_to_binary(self):
        # Load the result image.
        result = self.image

        # Convert the image to grayscale.
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Threshold the image to binarize it.
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Save the binary image.
        return thresh


if __name__ == "__main__":
    input_image_path = '/Users/vansh/PycharmProjects/Text-Extractor/text-image-processing/processed_image/result.png'
    output_binary_image_path = '/Users/vansh/PycharmProjects/Text-Extractor/text-image-processing/processed_image/result_binary.png'

    processor = Binarizer(cv2.imread(input_image_path))
    binary_image = processor.convert_to_binary()

    # Optionally, you can display the binary image
    cv2.imwrite(output_binary_image_path, binary_image)

