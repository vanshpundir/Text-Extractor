from img2table.ocr import TesseractOCR
from img2table.document import Image

# Instantiation of OCR
ocr = TesseractOCR(n_threads=1, lang="eng")

# Instantiation of document, either an image or a PDF
doc = Image("/Users/vansh/PycharmProjects/Text-Extractor/image_processing/8_image_without_lines_noise_removed.jpg")

# Table extraction
extracted_tables = doc.extract_tables(ocr=ocr,
                                      implicit_rows=False,
                                      borderless_tables=True,
                                      min_confidence=50)
print(extracted_tables[0].df)