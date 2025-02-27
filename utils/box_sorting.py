# import numpy as np

# def sort_boxes(boxes):
#     """Sort bounding boxes in reading order"""
#     boxes_array = np.array(boxes)
#     y_coords = boxes_array[:, 1]
#     sorted_indices = np.argsort(y_coords)
#     sorted_boxes = boxes_array[sorted_indices]

#     final_sorted = []
#     current_row = []
#     y_threshold = 20
#     current_y = None

#     for box in sorted_boxes:
#         x1, y1, x2, y2 = box
#         if current_y is None or abs(y1 - current_y) > y_threshold:
#             if current_row:
#                 current_row.sort(key=lambda b: b[0])  
#                 final_sorted.extend(current_row)
#             current_row = [box]
#             current_y = y1
#         else:
#             current_row.append(box)
    
#     if current_row:
#         current_row.sort(key=lambda b: b[0])
#         final_sorted.extend(current_row)

#     return [b.tolist() for b in final_sorted]


import numpy as np

def sort_boxes(boxes, row_threshold_ratio=0.5):
    """
    Sort bounding boxes in reading order (top-to-bottom, left-to-right)
    using the center y coordinate and a dynamic threshold based on box height.

    Args:
        boxes (list): List of bounding boxes in the format [x1, y1, x2, y2].
        row_threshold_ratio (float): Ratio to determine row grouping sensitivity.
                                     Lower values mean stricter grouping.
    
    Returns:
        List of boxes sorted in reading order.
    """
    # Compute center coordinates and height for each box
    boxes_info = []
    for box in boxes:
        x1, y1, x2, y2 = box
        center_y = (y1 + y2) / 2
        center_x = (x1 + x2) / 2
        height = y2 - y1
        boxes_info.append((box, center_y, center_x, height))
    
    # Sort boxes by vertical center (top-to-bottom)
    boxes_info.sort(key=lambda b: b[1])
    
    rows = []
    current_row = []
    current_row_center = None

    for info in boxes_info:
        box, center_y, center_x, height = info
        if not current_row:
            current_row = [info]
            current_row_center = center_y
        else:
            # Determine dynamic threshold using the average height of the current row
            avg_height = np.mean([b[3] for b in current_row])
            if abs(center_y - current_row_center) <= row_threshold_ratio * avg_height:
                # Box is in the same row; update row center
                current_row.append(info)
                current_row_center = np.mean([b[1] for b in current_row])
            else:
                # Sort current row by horizontal center and add to rows list
                current_row.sort(key=lambda b: b[2])
                rows.append([b[0] for b in current_row])
                # Start a new row
                current_row = [info]
                current_row_center = center_y

    if current_row:
        current_row.sort(key=lambda b: b[2])
        rows.append([b[0] for b in current_row])

    # Flatten the list of rows into a single list of boxes
    sorted_boxes = [box for row in rows for box in row]
    return sorted_boxes
