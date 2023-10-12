import cv2
import numpy as np
import matplotlib.pyplot as plt


class CropImage:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
        self.cropped_image = None

    def crop_image_to_largest_contour(self):
        """Crops the image to the bounding rectangle of the largest contour.

        Returns:
            The cropped image as a NumPy array.
        """

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = cv2.getRectSubPix(self.image, (w, h), (x + w // 2, y + h // 2))

        self.cropped_image = cropped_image

        return cropped_image

    def display_cropped_image(self):
        plt.imshow(cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2RGB))
        plt.show()


def main():
    image_path = '../images/extract_data/IMG_9518.jpg'

    image_processor = CropImage(image_path)
    image_processor.crop_image_to_largest_contour()
    image_processor.display_cropped_image()


if __name__ == '__main__':
    main()
