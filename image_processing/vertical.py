from PIL import Image
import pytesseract

class VerticalToHorizontal:
    def __init__(self, image_path):
        self.image = Image.open(image_path)
        self.orientation = self._check_orientation()

    def _check_orientation(self):
        text = pytesseract.image_to_osd(self.image)
        lines = text.splitlines()
        for line in lines:
            if "Orientation in degrees" in line:
                detected_orientation = float(line.split(":")[1].strip())
                return detected_orientation
        return 0

    def rotate(self):
        if self.orientation == 0:
            return self.image  # Image is already horizontal

        if self.orientation > 0:
            rotated_img = self.image.rotate(90, expand=True)
        else:
            rotated_img = self.image.rotate(-90, expand=True)

        if rotated_img.mode == 'RGBA':
            rotated_img = rotated_img.convert('RGB')

        return rotated_img

    def save_as_jpeg(self, output_path):
        rotated_img = self.rotate()
        rotated_img.save(output_path, 'JPEG')

    def save_as_png(self, output_path):
        rotated_img = self.rotate()
        rotated_img.save(output_path, 'PNG')

# Example usage:
if __name__ == '__main__':
    image_path = "/images/extract_data/test.png"
    converter = VerticalToHorizontal(image_path)
    converter.save_as_jpeg('rotated.jpg')
