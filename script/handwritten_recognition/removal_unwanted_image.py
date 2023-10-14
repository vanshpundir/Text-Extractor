from PIL import Image
import os

list1 = []

dir = '/Users/vansh/PycharmProjects/Text-Extractor/script/segmented_images'

# Iterate through files in the directory
for i in os.listdir(dir):
    if i.startswith('digit'):
        # Open the image using PIL
        try:
            with Image.open(os.path.join(dir, i)) as img:
                # Check if the image is not empty (has non-zero dimensions)
                if img.size[0] > 25 and img.size[1] > 45:
                    list1.append(i)
                else:
                    # Delete the image if it's empty
                    os.remove(os.path.join(dir, i))
        except Exception as e:
            # Handle any exceptions that may occur when opening the image
            print(f"Error processing {i}")

# Now, list1 contains the paths of non-empty images that start with 'digit'
