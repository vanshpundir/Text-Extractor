import cv2
import numpy as np

def extract_vertical_lines(image):
  """Extracts vertical lines from an image.

  Args:
    image: A numpy array representing the image.

  Returns:
    A list of contours representing the vertical lines.
  """

  # Convert the image to grayscale.
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply Canny edge detection.
  edges = cv2.Canny(gray, 50, 150)

  # Create a vertical structuring element.
  vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,1))

  # Dilate the edge image using the vertical structuring element.
  dilated_edges = cv2.dilate(edges, vertical_kernel)

  # Find the contours in the dilated image.
  contours, hierarchy = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Filter the contours to only keep the vertical lines.
  vertical_lines = []
  for contour in contours:
    # Calculate the aspect ratio of the contour.
    ar = cv2.boundingRect(contour)[2] / cv2.boundingRect(contour)[3]

    # If the aspect ratio is greater than 1, then the contour is a vertical line.
    if ar > 1:
      vertical_lines.append(contour)

  return vertical_lines

def draw_lines(image, lines):
  """Draws the lines on the image.

  Args:
    image: A numpy array representing the image.
    lines: A list of contours representing the lines.
  """

  # Draw the lines on the image.
  for line in lines:
    cv2.drawContours(image, [line], -1, (0, 0, 255), 2)

if __name__ == '__main__':
  # Load the image.
  image = cv2.imread('/Users/cup72/PycharmProjects/Text-Extractor/images/extract_data/IMG_9518.jpg')

  # Extract the vertical lines from the image.
  vertical_lines = extract_vertical_lines(image)

  # Draw the vertical lines on the image.
  draw_lines(image, vertical_lines)

  # Display the image.
  cv2.imshow('Image', image)
  cv2.waitKey(0)
