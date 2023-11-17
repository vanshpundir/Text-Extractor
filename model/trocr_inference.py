from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

class TrOCRInference:
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')

    def perform_ocr(self, image_path):
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

if __name__ == "__main__":
    # Replace the image_path with your specific image file path
    image_path = "/Users/vansh/PycharmProjects/Text-Extractor/image_processing_text/processed_image/horizontal_image/section_7.png"

    ocr_instance = TrOCRInference()
    generated_text = ocr_instance.perform_ocr(image_path)

    print("Generated Text:", generated_text)
