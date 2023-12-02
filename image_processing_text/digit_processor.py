import cv2
import os

class DigitProcessor:
    def __init__(self, input_directory, output_directory):
        self.input_directory = input_directory
        self.output_directory = output_directory

    def clear_output_directory(self):
        # Clear the output directory
        for file_name in os.listdir(self.output_directory):
            file_path = os.path.join(self.output_directory, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    def process_image(self, image_path):
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary image
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize containers for left and right digit images
        left_digit_image = None
        right_digit_image = None

        # Define the minimum width and height for contours
        min_width = 10
        min_height = 10

        # Separate the digits
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter out small contours
            if w > min_width and h > min_height:
                # Calculate the center of the bounding box
                center_x = x + w // 2

                # Check the x-coordinate to determine if it's the left or right digit
                if center_x < image.shape[1] / 2:
                    left_digit_image = image[y:y + h, x:x + w]
                else:
                    right_digit_image = image[y:y + h, x:x + w]

        # Save the left and right digit images if they exist
        if left_digit_image is not None:
            left_digit_path = os.path.join(self.output_directory, 'left_digit_' + os.path.basename(image_path))
            cv2.imwrite(left_digit_path, left_digit_image)

        if right_digit_image is not None:
            right_digit_path = os.path.join(self.output_directory, 'right_digit_' + os.path.basename(image_path))
            cv2.imwrite(right_digit_path, right_digit_image)

    def process_images_in_directory(self):

        self.clear_output_directory()
        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        # Iterate over the images in the input directory
        for filename in os.listdir(self.input_directory):
            if filename.endswith(".png"):  # Process only PNG files
                image_path = os.path.join(self.input_directory, filename)
                self.process_image(image_path)

        print("Processing completed.")

if __name__ == '__main':
    # Input and output directories
    input_directory = '/Users/vansh/PycharmProjects/Text-Extractor/image_processing_text/processed_image/horizontal_digit'
    output_directory = '/Users/vansh/PycharmProjects/Text-Extractor/image_processing_text/single_digit_processed'

    # Create an instance of the DigitProcessor class and process the images
    digit_processor = DigitProcessor(input_directory, output_directory)
    digit_processor.process_images_in_directory()
