from img2table.ocr import PaddleOCR

ocr = PaddleOCR(lang="en")

from img2table.document import Image

# Instantiation of document, either an image or a PDF
doc = Image("/home/shivam/Documents/Github/Text-Extractor/image_processing/rotated.jpg")

# Table extraction
extracted_tables = doc.extract_tables(ocr=ocr,
                                      implicit_rows=False,
                                      borderless_tables=False,
                                      min_confidence=50)

print(extracted_tables[0].df)