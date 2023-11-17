import numpy as np
import cv2
from entity.image import  Image
class Denoiser:
    def __init__(self,image,output_image_path):
        self.image = Image()
        self.image.array = image
        self.output_image_path = output_image_path

    def denoise_image(self):
        # Load the binary image.
        binary_image = self.image.array

        # Apply a median blur to the image.
        denoised_image = cv2.medianBlur(binary_image, 3)

        cv2.imwrite(self.output_image_path, denoised_image)
        return denoised_image


if __name__ == "__main__":
    input_path = cv2.imread('/Users/vansh/PycharmProjects/Text-Extractor/image_processing_text/processed_image/result_image.jpg')
    output_path = '/Users/vansh/PycharmProjects/Text-Extractor/image_processing_text'

    denoiser = Denoiser(input_path, output_path)
    cv2.imshow("image",denoiser.denoise_image())
    cv2.waitKey()
    cv2.destroyAllWindows()
