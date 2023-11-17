import cv2
import numpy as np


class HorizontalSectionExtractor:
    def __init__(self, image, output_dir, threshold=100):
        self.array = image
        self.output_dir = output_dir
        self.threshold = threshold

    def extract_sections(self):
        # Read the image
        image = self.array

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Initialize variables to track the start and end of each section
        start_row = None
        end_row = None

        # List to store the extracted sections
        sections = []

        # Iterate over the rows of the image
        for row in range(gray.shape[0]):
            # Check if the row contains white pixels above the threshold
            if np.sum(gray[row] > self.threshold) > 0:
                if start_row is None:
                    # This is the start of a new section
                    start_row = row
            else:
                if start_row is not None:
                    # This is the end of the current section
                    end_row = row
                    section = gray[start_row:end_row, :]
                    sections.append(section)
                    start_row = None
                    end_row = None

        # Save each section as a separate image
        for i, section in enumerate(sections):
            filename = f"{self.output_dir}/section_{i}.png"
            cv2.imwrite(filename, section)


if __name__ == "__main__":
    image = cv2.imread("/Users/vansh/PycharmProjects/Text-Extractor/image_processing_text/processed_image/result_denoised.jpg")
    output_directory = "/Users/vansh/PycharmProjects/Text-Extractor/image_processing_text/processed_image/horizontal_digit"

    section_extractor = HorizontalSectionExtractor(image, output_directory)
    section_extractor.extract_sections()
