import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


class ImageProcessor:
    def __init__(self, image_path, model_path):
        self.image_path = image_path
        self.model_path = model_path
        self.image = cv2.imread(self.image_path)
        self.model = load_model(self.model_path)

    def centralize(self, img):
        _, thresholded = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        coords = cv2.findNonZero(thresholded)
        x, y, width, height = cv2.boundingRect(coords)

        crop_x = x
        crop_y = y
        crop_width = width
        crop_height = height

        cropped_image = thresholded[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
        return cropped_image

    def extract_phone_number(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

        phone_number = ""
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w < 10 or h < 10:
                continue

            character = gray[y:y+h, x:x+w]
            character = self.centralize(character)

            resized_image = cv2.resize(character, (28, 28), interpolation=cv2.INTER_LINEAR)
            x_pred = resized_image.reshape(28, 28, 1)

            output = self.model.predict(np.array([x_pred]))
            output = np.argmax(output)

            phone_number += str(output)

        return phone_number


def main():
    image_path = 'try01.jpg'
    model_path = 'cnn-mnist-new.h5'

    image_processor = ImageProcessor(image_path, model_path)
    phone_number = image_processor.extract_phone_number()

    print(phone_number)


if __name__ == '__main__':
    main()
