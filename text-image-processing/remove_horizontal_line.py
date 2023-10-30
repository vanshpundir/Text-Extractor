import cv2
import numpy as np


class RemoveHorizontalLine:
    def __init__(self, image_path, horizontal_size, vertical_size, kernel_threshold):
        self.image = cv2.imread(image_path)
        self.horizontal_size = horizontal_size
        self.vertical_size = vertical_size
        self.kernel_threshold = kernel_threshold
        self.result_image = None  # Initialize result_image
        self.num_horizontal_lines = 0
        self.num_vertical_lines = 0

    def preprocess_image(self, threshold=0):
        # Convert the image to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary image with text in white
        _, binary_text = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Create a horizontal kernel for morphological operations
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.horizontal_size, 1))

        # Apply morphological operations to extract horizontal lines
        horizontal_lines = cv2.morphologyEx(binary_text, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

        # Create a vertical kernel for morphological operations
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.vertical_size))

        # Apply morphological operations to extract vertical lines
        vertical_lines = cv2.morphologyEx(binary_text, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

        # Count the number of horizontal and vertical lines
        self.num_horizontal_lines = np.sum(horizontal_lines == 255)
        self.num_vertical_lines = np.sum(vertical_lines == 255)

        # Merge the binary text image with inverted horizontal and vertical lines
        self.result_image = cv2.bitwise_and(binary_text,
                                            cv2.bitwise_not(cv2.bitwise_or(horizontal_lines, vertical_lines)))

        # Perform erosion and dilation operations to denoise the image
        kernel = np.ones(self.kernel_threshold, np.uint8)
        self.result_image = cv2.erode(self.result_image, kernel, iterations=1)
        self.result_image = cv2.dilate(self.result_image, kernel, iterations=1)
        return self.result_image

    def save_processed_image(self, output_path, denoise_kernel_size=(3, 3), invert_colors=True):
        if self.result_image is not None:
            # Optionally invert colors
            if invert_colors:
                self.result_image = cv2.bitwise_not(self.result_image)

            # Save the preprocessed image
            cv2.imwrite(output_path, self.result_image)


if __name__ == "__main__":
    # Create an instance of the RemoveHorizontalLine class
    processor = RemoveHorizontalLine("/home/shivam/Documents/Github/Text-Extractor/images/handwritten_img/Screenshot 2023-10-14 at 10.48.19 AM.png",
                                     horizontal_size=197, vertical_size=165, kernel_threshold=(3, 3))

    # Preprocess the image
    processed_image = processor.preprocess_image(threshold=0)

    # Save the processed image with inverted colors
    processor.save_processed_image("result_image_new_inverted.jpg", invert_colors=True)

    # Get the number of horizontal and vertical lines
    num_horizontal_lines = processor.num_horizontal_lines
    num_vertical_lines = processor.num_vertical_lines

    print(f"Number of horizontal lines: {num_horizontal_lines}")
    print(f"Number of vertical lines: {num_vertical_lines}")
