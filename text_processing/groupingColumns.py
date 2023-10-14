def group_columns(bounding_boxes, x_threshold=10):
    # Create an empty list to store grouped columns
    grouped_columns = []

    # Sort the bounding boxes by their x_min (x1 coordinate)
    sorted_boxes = sorted(bounding_boxes, key=lambda box: box[0][0])

    current_column = [sorted_boxes[0]]

    for i in range(1, len(sorted_boxes)):
        # Check if the x1 coordinate of the current bounding box is within the threshold of the x2 coordinate of the last bounding box in the current column
        if sorted_boxes[i][0][0] - current_column[-1][2][0] <= x_threshold:
            current_column.append(sorted_boxes[i])
        else:
            grouped_columns.append(current_column)
            current_column = [sorted_boxes[i]]

    grouped_columns.append(current_column)

    return grouped_columns

# Example usage:
if __name__ == "__main__":
    # Example list of bounding boxes in your format
    bounding_boxes = [
        [[2590.0, 226.0], [2720.0, 226.0], [2720.0, 300.0], [2590.0, 300.0]],
        [[2572.0, 140.0], [2688.0, 154.0], [2680.0, 232.0], [2563.0, 217.0]],
        [[2559.0, 392.0], [2721.0, 400.0], [2717.0, 470.0], [2555.0, 462.0]],
        [[642.0, 575.0], [929.0, 575.0], [929.0, 630.0], [642.0, 630.0]]
    ]

    # Group bounding boxes into columns
    grouped_columns = group_columns(bounding_boxes, x_threshold=15)

    # Print the result
    for i, column in enumerate(grouped_columns):
        print(f"Column {i + 1}: {column}")
