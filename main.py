import os
from image_processing_text.handwritten_image_processor import HandWrittenImageProcessor
from last_column_extraction.last_column import LastColumn

import pandas as pd
from model.trocr_inference import TrOCRInference
import re
import cv2
import shutil
from model.inference import Inference
from models.extract_text import extractText


class Main:
    def __init__(self, file_path, text_path, image_path, image_dir, model_path):
        self.file_path = file_path
        self.image_dir = image_dir
        self.image_path = image_path
        self.ocr_inference = TrOCRInference()
        self.last_column = LastColumn(text_path, image_path)
        self.mnist_model = model_path
        self.model = Inference(self.mnist_model)

    def get_last_column(self):
        bbox, cropped_image = self.last_column.extract_last_column()
        save_image = "cropped_image.jpg"
        cv2.imwrite(save_image, cropped_image)

        return cropped_image, save_image

    def horizontal_image(self):

        image, image_path = self.get_last_column()

        processor = HandWrittenImageProcessor(image_path)

        # Apply the processing steps
        processor.apply_add_white_line()
        processor.apply_binarize()
        processor.apply_denoiser("image_processing_text/result_denoised.jpg")
        processor.apply_horizontal_section_extractor(
            "image_processing_text/result_denoised.jpg",
            'image_processing_text/processed_image/horizontal_image')
        processor.apply_digit_processor('image_processing_text/processed_image/horizontal_image',
                                        'image_processing_text/processed_image/single_digit')

    def mnist_result(self, single_digit='image_processing_text/processed_image/single_digit'):
        self.model.mnist_result(single_digit)


    def extract_last_column(self):
        # Read the Excel file
        df = pd.read_excel(self.file_path)

        # Get the last column
        last_column = df.iloc[:, -1]

        return last_column

    @staticmethod
    def numerical_sort(value):
        parts = value.split('_')
        try:
            return int(parts[1].split('.')[0])  # Extract numeric part and remove file extension
        except (IndexError, ValueError):
            # Handle cases where the expected pattern is not found or conversion to int fails
            return float('inf')

    def get_two_digit_number(self):
        last_column = self.extract_last_column()
        numbers = []
        image_files = []
        for i in os.listdir(image_dir):
            if i.endswith(".png") or i.endswith(".jpg"):
                image_files.append(i)
        image_files = sorted(image_files, key=self.numerical_sort)
        print(image_files)

        # Get a list of image files ending with .png, sorted

        print("IMage file array is:", image_files)
        for filename in image_files[1:]:  # Skip the first image
            image_path = os.path.join(self.image_dir, filename)
            print(image_path)

            # Check if the image file exists
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                numbers.append(None)
                continue

            # Attempt to read the image using OpenCV
            try:
                img = cv2.imread(image_path)
                if img is None:
                    raise Exception("Error reading image using OpenCV.")
            except Exception as e:
                print(f"Error reading image: {image_path}. {e}")
                numbers.append(None)
                continue

            digit_text = self.ocr_inference.perform_ocr(image_path)
            print("digit text is: ", digit_text)
            digits = re.findall(r'\d+', digit_text)

            if len("".join(digits)) == 2:
                numbers.append(int("".join(digits)))
            else:
                numbers.append(None)

        return numbers

    def decider(self):
        df = pd.read_excel(self.file_path)
        last_column = self.extract_last_column()
        numbers = self.get_two_digit_number()

        gain =  0
        for i in range(len(last_column)):
            # Check if the horizontal image has valid information
            if last_column[i] == numbers[i]:
                print("Correct")
                gain += 1
            elif last_column[i] is None:
                last_column[i] = numbers[i]
        df[:-1] = last_column
        print("ACCURACY IS :", (gain/len(numbers)) * 100)
        return df

    def delete_small_height_images(self, height_threshold=10):
        """
        Delete images with a height less than the specified threshold in the given directory.

        Parameters:
        - height_threshold (int): Minimum height threshold for images to be deleted (default is 10 pixels).
        """
        # Ensure the directory exists
        directory_path = self.image_dir
        if not os.path.exists(directory_path):
            print(f"Directory '{directory_path}' not found.")
            return

        # Iterate through files in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            # Check if it's a file
            if os.path.isfile(file_path):
                # Read the image using OpenCV
                img = cv2.imread(file_path)

                # Check if the image is successfully read
                if img is not None:
                    # Check if the height is less than the threshold
                    if img.shape[0] < height_threshold:
                        # Delete the image file
                        os.remove(file_path)
                        print(f"Deleted: {filename} (Height: {img.shape[0]} pixels)")
                else:
                    print(f"Error reading image: {filename}")

        print("Deletion process completed.")


if __name__ == "__main__":
    # Replace the file path, image_dir, and model_path with your specific paths
    full_image = 'output/rotated/[1, 37, 2904, 857]_0.jpg'
    extractText(full_image)
    file_path = "script/output/rotated/[2, 2, 2902, 846]_0.xlsx"
    image_dir = "image_processing_text/processed_image/horizontal_image"
    model_path = "model/mnist_model_final.h5"
    text_path = "output/rotated/res_0.txt"
    image_path = "output/rotated/[1, 37, 2904, 857]_0.jpg"
    # Create an instance of the Main class and call the get_two_digit_number method
    main_instance = Main(file_path, text_path, image_path=image_path, image_dir=image_dir, model_path=model_path)
    main_instance.horizontal_image()
    main_instance.delete_small_height_images(20)
    # main_instance.mnist_result()
    # print(main_instance.mnist_result())
    main_instance.decider()
