import tensorflow as tf
import cv2
import numpy as np

class Inference:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def preprocess_image(self, image_path):
        custom_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        custom_image = cv2.resize(custom_image, (28, 28))
        custom_image = cv2.threshold(custom_image, 200, 255, cv2.THRESH_BINARY)[1]
        custom_image = custom_image.reshape(1, 28, 28, 1)
        return custom_image

    def predict(self, preprocessed_image):
        predictions = self.model.predict(preprocessed_image)
        predicted_digit = np.argmax(predictions)
        return predicted_digit

if __name__ =='__main__':
# Usage
    model_path = "mnist_model_final.h5"
    image_path = '/image_processing_text/image_processing_text/processed_image/single_digit/left_digit_section_1.png'

    digit_predictor = Inference(model_path)
    preprocessed_image = digit_predictor.preprocess_image(image_path)
    predicted_digit = digit_predictor.predict(preprocessed_image)
    print("Predicted Digit:", predicted_digit)
