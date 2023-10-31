import cv2
import numpy as np

def process_image(input_image_path, output_image_path=None):
    # Load the image
    image = cv2.imread(input_image_path)

    # Check if the image was loaded successfully
    if image is None:
        print("Failed to load image.")
    else:
        # Create a callback function to display the pixel intensity at the mouse cursor position
        def callback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                # Create a black canvas to display the text
                text_canvas = np.zeros_like(image, dtype=np.uint8)

                # Get the pixel intensity at the mouse cursor position
                pixel_intensity = image[y, x]

                # Create a text to display pixel intensity
                text = f'Pixel Intensity: {pixel_intensity}'

                # Put the text in red on the black canvas
                font = cv2.FONT_HERSHEY_DUPLEX  # Use a different font
                cv2.putText(text_canvas, text, (10, 30), font, 1, (0, 0, 255), 2)

                # Combine the text canvas with the original image
                result_image = cv2.addWeighted(image, 1, text_canvas, 1, 0)

                # Show the image with the updated text
                cv2.imshow("Pixel Intensity", result_image)

                # Print the pixel intensity
                print(pixel_intensity)

        # Create a window for displaying the image
        cv2.namedWindow("Pixel Intensity")

        # Set the callback function
        cv2.setMouseCallback("Pixel Intensity", callback)

        # Display the image
        cv2.imshow("Pixel Intensity", image)

        # Wait for a key press to exit
        cv2.waitKey(0)

        # Destroy all windows
        cv2.destroyAllWindows()

# Example usage:
process_image("/Users/cup72/PycharmProjects/Text-Extractor/last-column-extraction/outptu.jpg")
