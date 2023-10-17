import cv2


class ImageCropper:
    def __init__(self, input_image_path, output_image_path, threshold=10):
        self.input_image_path = input_image_path
        self.output_image_path = output_image_path
        self.threshold = threshold

    def crop_image(self):
        # Load the image
        image = cv2.imread(self.input_image_path, cv2.IMREAD_GRAYSCALE)

        # Define the left and right boundaries
        left_boundary = 0
        right_boundary = image.shape[1]

        # Iterate from the left to find the first white pixel
        for col in range(image.shape[1]):
            if 255 in image[:, col]:
                left_boundary = max(0, col - self.threshold)
                break

        # Iterate from the right to find the first white pixel
        for col in range(image.shape[1] - 1, -1, -1):
            if 255 in image[:, col]:
                right_boundary = min(image.shape[1], col + self.threshold)
                break

        # Crop the image with the specified amount of black region
        cropped_image = image[:, left_boundary:right_boundary]

        # Save the cropped image
        cv2.imwrite(self.output_image_path, cropped_image)


if __name__ == "__main__":
    input_path = '/Users/vansh/PycharmProjects/Text-Extractor/text-image-processing/processed_image/single_digit/right_image_8.png'
    output_path = "/Users/vansh/PycharmProjects/Text-Extractor/text-image-processing/processed_image/single_digit/crop1.jpg"

    image_cropper = ImageCropper(input_path, output_path)
    image_cropper.crop_image()
