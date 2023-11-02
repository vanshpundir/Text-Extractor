import os
import shutil

# Source directory containing the subdirectories with images
source_dir = '/Users/vansh/PycharmProjects/Text-Extractor/data/All Images'

# Destination directory to copy all images into
destination_dir = os.path.join(source_dir, 'image')

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Walk through the source directory and its subdirectories
for root, _, files in os.walk(source_dir):
    for file in files:
        source_path = os.path.join(root, file)
        destination_path = os.path.join(destination_dir, file)

        # Check if the destination file already exists and rename if necessary
        if os.path.exists(destination_path):
            base, ext = os.path.splitext(file)
            index = 1
            while os.path.exists(destination_path):
                new_filename = f"{base}_{index}{ext}"
                destination_path = os.path.join(destination_dir, new_filename)
                index += 1

        # Copy the image to the destination directory
        shutil.copy(source_path, destination_path)
        print(f'Copied: {source_path} to {destination_path}')

print('Image copy process completed.')
