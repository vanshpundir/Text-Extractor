import json

with open("/Users/vansh/PycharmProjects/Text-Extractor/output/rotated/res_0.txt") as f:
    for line in f:
        data = json.loads(line)
        break

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

print(f"X-Min: {x_min}")
print(f"X-Max: {x_max}")
print(f"Y-Min: {y_min}")
print(f"Y-Max: {y_max}")
