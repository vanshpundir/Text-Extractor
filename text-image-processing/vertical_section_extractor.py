import cv2

class AddWhiteLine:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)

    def process_image(self):
        # Create a white line
        white_line = np.zeros((self.image.shape[0], 1, 3), np.uint8)
        white_line[:, :, :] = 255

        # Add the white line to the image
        self.image[:, :, 0] = white_line[:, :, 0]
        self.image[:, :, 1] = white_line[:, :, 1]
        self.image[:, :, 2] = white_line[:, :, 2]

        return self.image

    def show_image(self):
        processed_image = self.process_image()
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    input_image_path = '/Users/vansh/PycharmProjects/Text-Extractor/images/handwritten_img/Screenshot 2023-10-14 at 10.48.19 AM.png'

    image_processor = AddWhiteLine(input_image_path)
    image_processor.show_image()
