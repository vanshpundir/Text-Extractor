from PIL import Image
import pytesseract

class ImageOrientationChecker:
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

    def check_orientation(self):
        return self.orientation == 0
