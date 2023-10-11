from paddleocr import PaddleOCR

# Initialize PaddleOCR with English language support and angle classification
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Path to the image you want to perform OCR on
img_path = '/Users/vansh/PycharmProjects/Text-Extractor/images/extract_data/IMG_9518.jpg'

# Perform OCR on the image
result = ocr.ocr(img_path, cls=True)

# Iterate through the results and print the detected text lines
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)
