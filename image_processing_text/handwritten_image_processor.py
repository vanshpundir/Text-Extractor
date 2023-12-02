from image_processing_text.add_white_line import AddWhiteLine
from image_processing_text.binarize import Binarizer
from image_processing_text.denoiser import Denoiser
from image_processing_text.horizontal_section_extractor import HorizontalSectionExtractor
from image_processing_text.image_cropper import ImageCropper
from entity.image import Image
from image_processing_text.digit_processor import DigitProcessor

import cv2
import os
import numpy as np

class HandWrittenImageProcessor:
    def __init__(self, input_image_path):
        self.input_image_path = input_image_path
        self.image = Image()
        self.image.array = None

    def apply_add_white_line(self):
        obj = AddWhiteLine(self.input_image_path)
        self.image.array = obj.process_image()

    def apply_binarize(self):
        obj = Binarizer(self.image.array)
        self.image.array = obj.convert_to_binary()

    def apply_denoiser(self, output_image_path):
        obj = Denoiser(self.image.array, output_image_path)
        self.image.array = obj.denoise_image()

    def apply_horizontal_section_extractor(self, image_path, output_dir):
        self.image.array = cv2.imread(image_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        obj = HorizontalSectionExtractor(self.image.array, output_dir)
        obj.extract_sections()

    def apply_digit_processor(self, horizontal_path_dir,
                              output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        digits = DigitProcessor(horizontal_path_dir, output_dir)
        digits.process_images_in_directory()
    def add_padding(self,input_dir, output_dir, padding_size=100):
        # Ensure the output directory exists, create it if necessary
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # List all files in the input directory
        input_files = os.listdir(input_dir)

        for file_name in input_files:
            # Construct the full path for both input and output
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            # Read the image using cv2
            image = cv2.imread(input_path)

            if image is not None:
                # Get the dimensions of the original image
                height, width, _ = image.shape

                # Calculate the new dimensions for the padded image
                new_width = width + 2 * padding_size
                new_height = height + 2 * padding_size

                # Create a new blank white image with the desired size
                padded_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                padded_image.fill(0)  # Fill with white color

                # Paste the original image onto the padded image with the padding
                padded_image[padding_size:padding_size + height, padding_size:padding_size + width] = image

                # Save the padded image to the output directory
                cv2.imwrite(output_path, padded_image)



    def apply_cropping(self, input_dir, output_dir):
        filenames = os.listdir(input_dir)
        left_images = [file for file in filenames if file.startswith("left_image")]
        right_images = [file for file in filenames if file.startswith("right_image")]

        for left_image in left_images:
            # Check if there is a corresponding right image
            right_image = left_image.replace("left_image", "right_image")
            if right_image in right_images:
                left_path = os.path.join(input_dir, left_image)
                right_path = os.path.join(input_dir, right_image)

                left_cropper = ImageCropper(left_path, os.path.join(output_dir, left_image))
                right_cropper = ImageCropper(right_path, os.path.join(output_dir, right_image))

                left_cropper.crop_image()
                right_cropper.crop_image()

if __name__ == "__main__":
        input_image_path = f'images/extract_data/img.png'
        processor = HandWrittenImageProcessor(input_image_path)

        # Apply the processing steps
        processor.apply_add_white_line()
        processor.apply_binarize()
        processor.apply_denoiser("image_processing_text/processed_image")
        processor.apply_horizontal_section_extractor("image_processing_text/processed_image/result_denoised.jpg",'image_processing_text/processed_image/horizontal_image')
        #processor.apply_digit_processor("/Users/vansh/PycharmProjects/Text-Extractor/image_processing_text/image_processing_text/processed_image/horizontal_image","/Users/vansh/PycharmProjects/Text-Extractor/final_images")
        #$processor.add_padding("/Users/vansh/PycharmProjects/Text-Extractor/image_processing_text/processed_image/single_digit",'/Users/vansh/PycharmProjects/Text-Extractor/image_processing_text/processed_image/single_digit_final',4)