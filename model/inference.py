import tensorflow as tf
import cv2
import numpy as np
import os

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

    def mnist_result(self, single_digit='image_processing_text/processed_image/single_digit'):


        file_value = {}
        for file in os.listdir(single_digit):
            if file.endswith(".png"):
                img = self.preprocess_image(os.path.join(single_digit, file))
                res = self.predict(img)
                file_value[file] = res
        return file_value

if __name__ =='__main__':
# Usage
    model_path = "mnist_model_final.h5"
    image_path = 'image_processing_text/processed_image/single_digit/left_digit_section_5.png'

    digit_predictor = Inference(model_path)
    print(digit_predictor.mnist_result())
