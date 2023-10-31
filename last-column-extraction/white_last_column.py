import cv2
import numpy as np

def process_image(input_image_path, output_image_path=None, threshold=80):
    # Load the image
    image = cv2.imread(input_image_path)

    # Check if the image was loaded successfully
    if image is None:
        print("Failed to load image.")
    else:
        # Iterate over the image and modify pixel values
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Check pixel intensity
                if np.all(image[i, j] > threshold):
                    image[i, j] = 255

        # Save the modified image if an output path is provided
        if output_image_path:
            cv2.imwrite(output_image_path, image)

        # Display the modified image (optional)
        cv2.imshow("Modified Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

process_image("/Users/vansh/PycharmProjects/Text-Extractor/images/handwritten_img/Screenshot 2023-10-14 at 10.48.19 AM.png","/Users/vansh/PycharmProjects/Text-Extractor/last-column-extraction/white_image.png",115)