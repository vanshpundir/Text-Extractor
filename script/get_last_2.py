import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Image path
image_path = "/output/rotated/[1, 37, 2904, 857]_0.jpg"

# Bounding box coordinates
bbox = [[2741.0, 0.0], [2756.0, 479.0], [2756.0, 809.0], [2741.0, 809.0]] # added 0 in the [0][0][1] position mannually

# Read the image
image = cv2.imread(image_path)

# Display the image
fig, ax = plt.subplots(1)
ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Add width only to the left side of the bounding box
width = 220  # Adjust the width as needed
bbox[0][0] -= width
bbox[3][0] -= width

# Create a Rectangle patch
rect = patches.Rectangle((bbox[0][0], bbox[0][1]), bbox[1][0] - bbox[0][0], bbox[2][1] - bbox[0][1], linewidth=2, edgecolor='r', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

# Show the image with the annotated bounding box
plt.show()
