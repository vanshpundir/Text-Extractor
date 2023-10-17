from add_white_line import AddWhiteLine
from binarize import Binarizer
from denoiser import Denoiser
from horizontal_section_extractor import HorizontalSectionExtractor
from vertical_image_splitter import VerticalImageSplitter
from image_cropper import ImageCropper
from entity.image import Image

import cv2
import os
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

    def apply_denoiser(self, output_dir="/Users/vansh/PycharmProjects/Text-Extractor/text-image-processing/processed_image"):
        obj = Denoiser(self.image.array, output_dir)
        self.image.array = obj.denoise_image()

    def apply_horizontal_section_extractor(self, image_path ,output_dir="/Users/vansh/PycharmProjects/Text-Extractor/text-image-processing/processed_image/horizontal_digit"):
        self.image.array = cv2.imread(image_path)
        obj = HorizontalSectionExtractor(self.image.array, output_dir)
        obj.extract_sections()

    def apply_vertical_image_splitter(self, horizontal_path_dir, output_dir="text-image-processing/processed_image/single_digit"):
        for index, i in enumerate(os.listdir(horizontal_path_dir)):
            filename = os.path.join(horizontal_path_dir, i)
            self.image.array = cv2.imread(filename)
            obj = VerticalImageSplitter(self.image.array, output_dir, index)
            obj.split_image()

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
    input_image_path = '/Users/vansh/PycharmProjects/Text-Extractor/images/handwritten_img/Screenshot 2023-10-14 at 10.48.19 AM.png'
    processor = HandWrittenImageProcessor(input_image_path)

    # Apply the processing steps
    processor.apply_add_white_line()
    processor.apply_binarize()
    processor.apply_denoiser()
    processor.apply_horizontal_section_extractor("text-image-processing/processed_image/result_denoised.png")
    processor.apply_vertical_image_splitter("/Users/vansh/PycharmProjects/Text-Extractor/text-image-processing/processed_image/horizontal_digit")
    processor.apply_cropping("text-image-processing/processed_image/single_digit","/Users/vansh/PycharmProjects/Text-Extractor/text-image-processing/processed_image/cropped_image")