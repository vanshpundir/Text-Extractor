import numpy as np
import cv2
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt

# Load the MNIST dataset
(x_train, _), _ = mnist.load_data()

# Choose an example MNIST image
image = cv2.imread('/Users/cup72/PycharmProjects/Text-Extractor/image_processing_text/image_processing_text/processed_image/single_digit_processed/left_digit_section_44.png')  # Change the index to the image you want to process

# Define a structuring element for morphological operations
kernel = np.ones((4, 4), np.uint8)

# Perform erosion to reduce white thickness
eroded_image = cv2.erode(image, kernel, iterations=1)

# Display the original and eroded images
cv2.imwrite('/Users/cup72/PycharmProjects/Text-Extractor/image_processing_text/image_processing_text/processed_image/single_digit_processed/left_digit_section_44.png',eroded_image)
