from PIL import Image
import pytesseract
# Read the image using Pillow
img = Image.open("/images/extract_data/test.png")

# Check the orientation using Tesseract
text = pytesseract.image_to_osd(img)
lines = text.splitlines()

# Variable to track if a horizontal heading is found
horizontal_heading_found = False

# Iterate through the detected lines
for line in lines:
    if "Orientation in degrees" in line:
        detected_orientation = float(line.split(":")[1].strip())
    elif "Text: " in line:
        # Check if the text is horizontal
        if "Horizontal" in line:
            horizontal_heading_found = True

# If a horizontal heading is found, the image is likely already horizontal
if horizontal_heading_found:
    rotated_img = img
else:
    # Rotate the image by 90 degrees
    if detected_orientation > 0:
        rotated_img = img.rotate(90, expand=True)
    else:
        rotated_img = img.rotate(-90, expand=True)

# Ensure the image is in RGB mode (not RGBA) before saving as JPEG
if rotated_img.mode == 'RGBA':
    rotated_img = rotated_img.convert('RGB')

# Save the rotated image as JPEG
rotated_img.save('rotated.jpg', 'JPEG')
