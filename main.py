import os
import tensorflow as tf
import numpy as np
import cv2

def preprocess_image(image):
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Threshold the image to make it binary
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # No need to display the thresholded image here
    return thresholded_image

def predict_single_digit(model_path, image_path):
    # Load the pre-trained MNIST model
    model = tf.keras.models.load_model(model_path)

    if os.path.exists(image_path):
        # Load and preprocess the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Check if the image is loaded successfully
        if image is not None:
            image = cv2.resize(image, (28, 28))  # Resize the image to 28x28

            # Apply preprocessing
            preprocessed_image = preprocess_image(image)

            preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add a batch dimension

            # Perform inference for the preprocessed image
            predictions = model.predict(preprocessed_image)

            # Get the predicted class (digit)
            predicted_digit = np.argmax(predictions)

            return predicted_digit
        else:
            print(f"Failed to load image: {image_path}")
            return None
    else:
        print(f"Image not found: {image_path}")
        return None

# Usage example:
model_path = '/Users/vansh/Downloads/cnn-mnist-new (1).h5'  # Replace with your model path
image_dir = '/Users/vansh/PycharmProjects/Text-Extractor/text-image-processing/processed_image/single_digit_final'

for image_filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_filename)
    predicted_digit = predict_single_digit(model_path, image_path)

    if predicted_digit is not None:
        print(f"The model predicts the digit in {image_filename} as: {predicted_digit}")
