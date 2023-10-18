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
        white_pixel_count = 0
        for col in range(image.shape[1]):
            if 255 in image[:, col]:
                white_pixel_count += 1
                if white_pixel_count >= self.threshold:
                    break
            else:
                white_pixel_count = 0
        left_boundary = max(0, col - self.threshold)

        # Iterate from the right to find the first white pixel
        white_pixel_count = 0
        for col in range(image.shape[1] - 1, -1, -1):
            if 255 in image[:, col]:
                white_pixel_count += 1
                if white_pixel_count >= self.threshold:
                    break
            else:
                white_pixel_count = 0
        right_boundary = min(image.shape[1], col + self.threshold)

        # Crop the image with the specified amount of black region
        cropped_image = image[:, left_boundary:right_boundary]

        # Save the cropped image
        cv2.imwrite(self.output_image_path, cropped_image)


if __name__ == "__main__":
    input_path = '/Users/vansh/PycharmProjects/Text-Extractor/text-image-processing/processed_image/single_digit/right_image12.png'
    output_path = "/Users/vansh/PycharmProjects/Text-Extractor/text-image-processing/processed_image/single_digit/crop1.jpg"

    image_cropper = ImageCropper(input_path, output_path)
    image_cropper.crop_image()
