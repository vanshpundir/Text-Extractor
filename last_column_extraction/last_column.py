import json
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class LastColumn:
    def __init__(self, text_file, image_file):
        self.text_file = text_file
        self.image = image_file

    def extract_last_column(self):
        with open(self.text_file) as f:
            for line in f:
                data = json.loads(line)
                break
        image = cv2.imread("output/rotated/[1, 37, 2904, 857]_0.jpg")
        # Assuming data['res'] contains a list of entries, each having a list of points in 'text_region'
        all_boxes = [point for entry in data['res'] for point in entry['text_region']]

        # Sort the bounding boxes based on X-coordinate
        sorted_boxes = sorted(all_boxes, key=lambda box: box[0])

        # Select the boxes with the highest X-coordinates as the last column
        last_column_boxes = sorted_boxes[-4:]  # Assuming each entry has 4 points

        # Calculate the bounding box for the last column
        x_min = min(point[0] for point in last_column_boxes)
        x_max = max(point[0] for point in last_column_boxes)
        y_min = min(point[1] for point in last_column_boxes)
        y_max = max(point[1] for point in last_column_boxes)

        bbox = [[x_min, 0.0], [x_max, y_min], [x_max, y_max],
                [x_min, y_max]]  # added 0 in the [0][0][1] position mannually

        # Read the image
        # Display the image


        # Add width only to the left side of the bounding box
        width = 220  # Adjust the width as needed
        bbox[0][0] -= width
        bbox[3][0] -= width


        # Crop the image based on the bounding box
        cropped_image = image[int(bbox[0][1]):int(bbox[2][1]), int(bbox[0][0]):int(bbox[2][0])]

        processed_image = self.add_white_background(cropped_image, 75)

        return bbox, processed_image

    def annotate_over_image(self):
        # Bounding box coordinates
        bbox, _ = self.extract_last_column()

        # Read the image
        # Display the image
        fig, ax = plt.subplots(1)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Create a Rectangle patch
        rect = patches.Rectangle((bbox[0][0], bbox[0][1]), bbox[1][0] - bbox[0][0], bbox[2][1] - bbox[0][1],
                                 linewidth=2, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

        # Show the image with the annotated bounding box
        plt.show()

    def add_white_background(self, image, threshold=75):

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

        return image


if __name__ == "__main__":
    last_column = LastColumn("output/rotated/res_0.txt", cv2.imread(
        "output/rotated/[1, 37, 2904, 857]_0.jpg"))
    bbox, cropped_image = last_column.extract_last_column()
    cv2.imshow("image", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
