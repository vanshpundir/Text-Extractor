import cv2

class RemoveVerticalLines:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)

    def process_image(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Define a vertical kernel for erosion and dilation
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        # Erosion to remove vertical lines
        removed_vertical_lines = cv2.erode(thresh, vertical_kernel, iterations=1)

        # Dilation to recover the remaining content
        result = cv2.dilate(removed_vertical_lines, vertical_kernel, iterations=1)

        return result

    def save_processed_image(self, output_path):
        processed_image = self.process_image()
        cv2.imwrite(output_path, processed_image)

if __name__ == "__main__":
    input_image_path = '/Users/vansh/PycharmProjects/Text-Extractor/images/handwritten_img/Screenshot 2023-10-14 at 10.48.19 AM.png'
    output_image_path = '/Users/vansh/PycharmProjects/Text-Extractor/text-image-processing/processed_image/result_vertical.png'

    image_processor = RemoveVerticalLines(input_image_path)
    image_processor.save_processed_image(output_image_path)
