import cv2

class RemoveVerticalLines:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)

    def process_image(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Remove vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(self.image, [c], -1, (255, 255, 255), 2)

        # Repair image
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        result = 255 - cv2.morphologyEx(255 - self.image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

        return result

    def save_processed_image(self, output_path):
        processed_image = self.process_image()
        cv2.imwrite(output_path, processed_image)

if __name__ == "__main__":
    input_image_path = '/Users/vansh/PycharmProjects/Text-Extractor/images/handwritten_img/Screenshot 2023-10-14 at 10.48.19 AM.png'
    output_image_path = '/Users/vansh/PycharmProjects/Text-Extractor/text-image-processing/processed_image/result_vertical.png'

    image_processor = RemoveVerticalLines(input_image_path)
    image_processor.save_processed_image(output_image_path)
