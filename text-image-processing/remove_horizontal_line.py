import cv2
import numpy as np

class RemoveHorizontalLine:
    def __init__(self, image_path, horizontal_size, vertical_size,kernel_threshold):
        self.image = cv2.imread(image_path)
        self.horizontal_size = horizontal_size
        self.vertical_size = vertical_size
        self.kernel_threshold = kernel_threshold
        self.result_image = None  # Initialize result_image

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

        # Merge the binary text image with inverted horizontal and vertical lines
        self.result_image = cv2.bitwise_and(binary_text, cv2.bitwise_not(cv2.bitwise_or(horizontal_lines, vertical_lines)))

        # Perform erosion and dilation operations to denoise the image
        kernel = np.ones(self.kernel_threshold, np.uint8)
        self.result_image = cv2.erode(self.result_image, kernel, iterations=1)
        self.result_image = cv2.dilate(self.result_image, kernel, iterations=1)
        return self.result_image

    def save_processed_image(self, output_path, denoise_kernel_size=(3, 3)):
        if self.result_image is not None:
            # Save the preprocessed image
            cv2.imwrite(output_path, self.result_image)
if __name__ == "__main__":
    # Create an instance of the ImageProcessor class
    #processor = RemoveHorizontalLine("/Users/cup72/PycharmProjects/Text-Extractor/images/handwritten_img/Screenshot 2023-10-14 at 10.48.19 AM.png", horizontal_size=95, vertical_size=65,kernel_threshold = (2,2))
    processor = RemoveHorizontalLine("/Users/cup72/PycharmProjects/Text-Extractor/images/handwritten_img/IMG_0037.jpg", horizontal_size=197, vertical_size=165,kernel_threshold = (3,3))
    # Preprocess the image
    processor.preprocess_image(threshold=0)

    # Save the processed images
    processor.save_processed_image("/Users/cup72/PycharmProjects/Text-Extractor/text-image-processing/text-image-processing/processed_image/result_image.jpg")

   #TODO:
   # if photo cells are small then kernel size will increase
   # Increasing hoizontal_size and vertical_size will make small digits visible