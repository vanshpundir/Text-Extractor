import os
from PIL import Image, ImageDraw


def get_bbox(bounding_boxes, x_threshold=0, y_threshold=0):
    grouped_columns = []
    remaining_boxes = list(bounding_boxes)

    while remaining_boxes:
        current_column = [remaining_boxes[0]]
        remaining_boxes.remove(remaining_boxes[0])

        i = 0
        while i < len(remaining_boxes):
            box = remaining_boxes[i]
            # Check if the box is within the x_threshold and y_threshold of the current column
            if abs(current_column[-1][2][0] - box[0][0]) <= x_threshold and abs(current_column[-1][0][1] - box[0][1]) <= y_threshold:
                current_column.append(box)
                remaining_boxes.remove(box)
            else:
                i += 1

        grouped_columns.append(current_column)

    return grouped_columns

def crop_columns(image, grouped_columns, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, column in enumerate(grouped_columns):
        # Get the coordinates of the current column
        x_min = min(box[0][0] for box in column)
        x_max = max(box[2][0] for box in column)
        y_min = min(box[0][1] for box in column)
        y_max = max(box[2][1] for box in column)

        # Crop the column from the original image
        cropped_column = image.crop((x_min, y_min, x_max, y_max))

        # Save the cropped column as an image
        column_filename = os.path.join(output_directory, f'column_{i + 1}.png')
        cropped_column.save(column_filename)

def main():
    data = {'type': 'figure', 'bbox': [1, 37, 2904, 857], 'res': [{'text': 'Sr. No.', 'confidence': 0.9407315254211426, 'text_region': [[117.0, 72.0], [295.0, 79.0], [293.0, 137.0], [115.0, 129.0]]}, {'text': 'Grade', 'confidence': 0.9905804395675659, 'text_region': [[2559.0, 66.0], [2732.0, 66.0], [2732.0, 123.0], [2559.0, 123.0]]}, {'text': 'Roll No.', 'confidence': 0.9969607591629028, 'text_region': [[392.0, 78.0], [603.0, 86.0], [601.0, 150.0], [390.0, 142.0]]}, {'text': 'Name', 'confidence': 0.9891712069511414, 'text_region': [[983.0, 91.0], [1139.0, 99.0], [1136.0, 160.0], [980.0, 151.0]]}, {'text': 'Email Id', 'confidence': 0.9054730534553528, 'text_region': [[1833.0, 82.0], [2062.0, 75.0], [2064.0, 133.0], [1835.0, 140.0]]}, {'text': '41', 'confidence': 0.6754592657089233, 'text_region': [[2572.0, 140.0], [2688.0, 154.0], [2680.0, 232.0], [2563.0, 217.0]]}, {'text': '2110993771', 'confidence': 0.9963628053665161, 'text_region': [[350.0, 168.0], [625.0, 179.0], [623.0, 236.0], [348.0, 225.0]]}, {'text': '1', 'confidence': 0.9937686920166016, 'text_region': [[179.0, 178.0], [216.0, 178.0], [216.0, 223.0], [179.0, 223.0]]}, {'text': 'Baashi Nazir', 'confidence': 0.9847583770751953, 'text_region': [[667.0, 178.0], [957.0, 185.0], [956.0, 242.0], [666.0, 235.0]]}, {'text': 'baashi3771.be21@chitkara.edu.in', 'confidence': 0.9967362880706787, 'text_region': [[1446.0, 178.0], [2199.0, 165.0], [2200.0, 229.0], [1447.0, 242.0]]}, {'text': '3', 'confidence': 0.8344473838806152, 'text_region': [[2590.0, 226.0], [2720.0, 226.0], [2720.0, 300.0], [2590.0, 300.0]]}, {'text': '2', 'confidence': 0.9984017014503479, 'text_region': [[167.0, 252.0], [210.0, 252.0], [210.0, 306.0], [167.0, 306.0]]}, {'text': '2110993795', 'confidence': 0.9986242055892944, 'text_region': [[341.0, 248.0], [621.0, 255.0], [620.0, 313.0], [339.0, 306.0]]}, {'text': 'Hitakshi', 'confidence': 0.9929057359695435, 'text_region': [[666.0, 258.0], [857.0, 258.0], [857.0, 316.0], [666.0, 316.0]]}, {'text': 'hitakshi3795.be21@chitkara.edu.in', 'confidence': 0.9876337647438049, 'text_region': [[1443.0, 258.0], [2232.0, 245.0], [2233.0, 303.0], [1444.0, 316.0]]}, {'text': '64', 'confidence': 0.9340137243270874, 'text_region': [[2565.0, 306.0], [2723.0, 306.0], [2723.0, 393.0], [2565.0, 393.0]]}, {'text': '3', 'confidence': 0.9968898892402649, 'text_region': [[158.0, 328.0], [201.0, 328.0], [201.0, 383.0], [158.0, 383.0]]}, {'text': '2110993811', 'confidence': 0.994162917137146, 'text_region': [[335.0, 322.0], [612.0, 329.0], [611.0, 393.0], [333.0, 386.0]]}, {'text': 'Mehakpreet', 'confidence': 0.9783431887626648, 'text_region': [[655.0, 334.0], [927.0, 342.0], [925.0, 399.0], [654.0, 392.0]]}, {'text': 'mehakpreet3811.be21@chitkara.edu.in', 'confidence': 0.978182852268219, 'text_region': [[1437.0, 335.0], [2323.0, 322.0], [2324.0, 386.0], [1438.0, 399.0]]}, {'text': '29', 'confidence': 0.9961168766021729, 'text_region': [[2559.0, 392.0], [2721.0, 400.0], [2717.0, 470.0], [2555.0, 462.0]]}, {'text': '4', 'confidence': 0.9986521601676941, 'text_region': [[152.0, 405.0], [192.0, 405.0], [192.0, 463.0], [152.0, 463.0]]}, {'text': '2110993832', 'confidence': 0.9976381063461304, 'text_region': [[328.0, 409.0], [609.0, 409.0], [609.0, 466.0], [328.0, 466.0]]}, {'text': 'Shivam Pandey', 'confidence': 0.9574026465415955, 'text_region': [[643.0, 408.0], [996.0, 415.0], [995.0, 479.0], [642.0, 472.0]]}, {'text': 'shivam3832.be21@chitkara.edu.in', 'confidence': 0.9884561896324158, 'text_region': [[1440.0, 415.0], [2223.0, 408.0], [2224.0, 466.0], [1441.0, 473.0]]}, {'text': '5', 'confidence': 0.9785011410713196, 'text_region': [[143.0, 485.0], [182.0, 485.0], [182.0, 546.0], [143.0, 546.0]]}, {'text': '82', 'confidence': 0.7185071110725403, 'text_region': [[2574.0, 479.0], [2756.0, 479.0], [2756.0, 559.0], [2574.0, 559.0]]}, {'text': '2110993839', 'confidence': 0.9976757168769836, 'text_region': [[319.0, 489.0], [597.0, 489.0], [597.0, 546.0], [319.0, 546.0]]}, {'text': 'SOHIL DHIMAN', 'confidence': 0.9621644616127014, 'text_region': [[645.0, 495.0], [1047.0, 495.0], [1047.0, 549.0], [645.0, 549.0]]}, {'text': 'sohil3839.be21@chitkara.edu.in', 'confidence': 0.9800654053688049, 'text_region': [[1437.0, 495.0], [2172.0, 495.0], [2172.0, 549.0], [1437.0, 549.0]]}, {'text': '6', 'confidence': 0.9861018657684326, 'text_region': [[128.0, 569.0], [179.0, 569.0], [179.0, 630.0], [128.0, 630.0]]}, {'text': '36', 'confidence': 0.9920941591262817, 'text_region': [[2562.0, 562.0], [2732.0, 562.0], [2732.0, 642.0], [2562.0, 642.0]]}, {'text': '2110993852', 'confidence': 0.9975963830947876, 'text_region': [[309.0, 572.0], [594.0, 572.0], [594.0, 626.0], [309.0, 626.0]]}, {'text': 'Vrinda Vritti', 'confidence': 0.9489549994468689, 'text_region': [[642.0, 575.0], [929.0, 575.0], [929.0, 630.0], [642.0, 630.0]]}, {'text': 'vrinda3852.be21@chitkara.edu.in', 'confidence': 0.9847683906555176, 'text_region': [[1434.0, 572.0], [2208.0, 572.0], [2208.0, 636.0], [1434.0, 636.0]]}, {'text': '7', 'confidence': 0.9981409311294556, 'text_region': [[125.0, 655.0], [167.0, 655.0], [167.0, 710.0], [125.0, 710.0]]}, {'text': '2110993858', 'confidence': 0.9736341238021851, 'text_region': [[300.0, 652.0], [588.0, 652.0], [588.0, 710.0], [300.0, 710.0]]}, {'text': 'ADVITIYA BHARTIGUPTA', 'confidence': 0.9566539525985718, 'text_region': [[630.0, 655.0], [1323.0, 658.0], [1322.0, 713.0], [630.0, 710.0]]}, {'text': '22', 'confidence': 0.9602898359298706, 'text_region': [[2571.0, 646.0], [2732.0, 646.0], [2732.0, 729.0], [2571.0, 729.0]]}, {'text': 'advitiya3858.be21@chitkara.edu.in', 'confidence': 0.986611545085907, 'text_region': [[1434.0, 665.0], [2245.0, 665.0], [2245.0, 710.0], [1434.0, 710.0]]}, {'text': '8', 'confidence': 0.9955339431762695, 'text_region': [[110.0, 735.0], [161.0, 735.0], [161.0, 796.0], [110.0, 796.0]]}, {'text': '2110993876', 'confidence': 0.9973891377449036, 'text_region': [[292.0, 732.0], [579.0, 739.0], [578.0, 797.0], [291.0, 789.0]]}, {'text': 'Jatin', 'confidence': 0.9898037910461426, 'text_region': [[609.0, 738.0], [736.0, 738.0], [736.0, 796.0], [609.0, 796.0]]}, {'text': '89', 'confidence': 0.817832350730896, 'text_region': [[2559.0, 729.0], [2741.0, 729.0], [2741.0, 809.0], [2559.0, 809.0]]}, {'text': 'jatin3876.be21@chitkara.edu.in', 'confidence': 0.9817367792129517, 'text_region': [[1428.0, 742.0], [2166.0, 742.0], [2166.0, 796.0], [1428.0, 796.0]]}], 'img_idx': 0}

    # Extract text regions and confidence scores from the data
    bounding_boxes = [region['text_region'] for region in data['res']]

    # Group the columns based on the x-coordinates of the bounding boxes
    bbox = get_bbox(bounding_boxes)

    # Load the image
    image_path = 'image_processing/rotated.jpg'
    image = Image.open(image_path)

    # Draw rectangles around the grouped columns
    draw = ImageDraw.Draw(image)
    for b_box in bbox:
        x_min = min(box[0][0] for box in b_box)
        x_max = max(box[2][0] for box in b_box)
        y_min = min(box[0][1] for box in b_box)
        y_max = max(box[2][1] for box in b_box)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

    # Save or display the modified image
    image.save('modified/image.jpg')
    image.show()
# Output directory to save cropped column images
    output_directory = 'final_images'

    # Crop and save individual columns as separate images
    crop_columns(image, bbox, output_directory)

if __name__ == "__main__":
    main()