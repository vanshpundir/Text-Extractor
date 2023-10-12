from PIL import Image
import pytesseract

class ImageOrientationHandler:
    def __init__(self, image_path):
        self.image = Image.open(image_path)
        self.detected_orientation = 0
        self.horizontal_heading_found = False

    def check_orientation(self):
        text = pytesseract.image_to_osd(self.image)
        lines = text.splitlines()

        for line in lines:
            if "Orientation in degrees" in line:
                self.detected_orientation = float(line.split(":")[1].strip())
            elif "Text: " in line and "Horizontal" in line:
                self.horizontal_heading_found = True

    def rotate_and_save(self, output_path='rotated.jpg'):
        if self.horizontal_heading_found:
            rotated_img = self.image
        else:
            if self.detected_orientation > 0:
                rotated_img = self.image.rotate(90, expand=True)
            else:
                rotated_img = self.image.rotate(-90, expand=True)

        if rotated_img.mode == 'RGBA':
            rotated_img = rotated_img.convert('RGB')

        rotated_img.save(output_path, 'JPEG')
