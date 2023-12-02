import cv2

class AddWhiteLine:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)

    def process_image(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Remove horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(self.image, [c], -1, (255, 255, 255), 2)

        # Repair image
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
        result = 255 - cv2.morphologyEx(255 - self.image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

        return result

    def save_processed_image(self, output_path):
        processed_image = self.process_image()
        cv2.imwrite(output_path, processed_image)

if __name__ == "__main__":
    input_image_path = 'images/handwritten_img/IMG_0037.jpg'
    output_image_path = '/text-image-processing/processed_image/result.png'

    image_processor = AddWhiteLine(input_image_path)
    image_processor.save_processed_image(output_image_path)
