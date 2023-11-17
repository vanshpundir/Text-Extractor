import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
# train_images = train_images / 255.0
# test_images = test_images / 255.0
import cv2
import numpy as np
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

class ImageCropper:
    def __init__(self, input_image, threshold=10):
        self.input_image = input_image
        self.threshold = threshold

    def crop_image(self):
        # Define the left and right boundaries
        left_boundary = 0
        right_boundary = self.input_image.shape[1]

        # Iterate from the left to find the first white pixel
        white_pixel_count = 0
        for col in range(self.input_image.shape[1]):
            if 255 in self.input_image[:, col]:
                white_pixel_count += 1
                if white_pixel_count >= self.threshold:
                    break
            else:
                white_pixel_count = 0
        left_boundary = max(0, col - self.threshold)

        # Iterate from the right to find the first white pixel
        white_pixel_count = 0
        for col in range(self.input_image.shape[1] - 1, -1, -1):
            if 255 in self.input_image[:, col]:
                white_pixel_count += 1
                if white_pixel_count >= self.threshold:
                    break
            else:
                white_pixel_count = 0
        right_boundary = min(self.input_image.shape[1], col + self.threshold)

        # Crop the image with the specified amount of black region
        cropped_image = self.input_image[:, left_boundary:right_boundary]

        # Check if the cropped_image is not empty
        if cropped_image.size == 0:
            return None

        # Resize the cropped image to 28x28
        cropped_image = cv2.resize(cropped_image, (28, 28))

        return cropped_image

# Iterate over the train_images and preprocess them
for i in range(len(train_images)):
    image = train_images[i]

    # Create an ImageCropper instance
    image_cropper = ImageCropper(image)

    # Crop the image
    cropped_image = image_cropper.crop_image()
    # Check if the cropped image is not empty
    if cropped_image is not None:
        # Set the cropped image as the new image
        train_images[i] = cropped_image

# Normalize pixel values to be between 0 and 1


# Build the CNN model
model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
model.save("/Users/vansh/PycharmProjects/Text-Extractor/image_processing_text/image_processing_text/processed_image/model/mnist_model_final.h5")