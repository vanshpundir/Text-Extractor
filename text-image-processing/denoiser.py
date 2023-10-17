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

        cv2.imwrite(f"{self.output_image_path}/result_denoised.jpg", denoised_image)
        return denoised_image


if __name__ == "__main__":
    input_path = cv2.imread('/Users/vansh/PycharmProjects/Text-Extractor/text-image-processing/processed_image/result_binary.png')
    output_path = 'processed_image/result_denoised.png'

    denoiser = Denoiser(input_path, output_path)
    denoiser.denoise_image()
