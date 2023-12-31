import pandas as pd
from paddleocr import PaddleOCR

class ExtractTypedText:
    def __init__(self, use_angle_cls=True, lang='en'):
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)

    def read_image(self, img_path):
        result = self.ocr.ocr(img_path, cls=True)
        for res in result:
            for line in res:
                print(line)
if __name__ == "__main__":
    ocr = ExtractTypedText()
    ocr.read_image('/Users/vansh/PycharmProjects/Text-Extractor/images/extract_data/image_text.png')