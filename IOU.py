def IOU(bbox, bbox_label):
    min_x1 = bbox[0]
    min_y1 = bbox[1]
    max_x1 = bbox[0]+bbox[2]
    max_y1 = bbox[1]+bbox[3]
    
    min_x2 = bbox_label[0]
    min_y2 = bbox_label[1]
    max_x2 = bbox_label[0]+bbox_label[2]
    max_y2 = bbox_label[1]+bbox_label[3]

    ret = 0
    # get area of rectangle A and B
    rect1_area = (max_x1 - min_x1) * (max_y1 - min_y1)
    rect2_area = (max_x2 - min_x2) * (max_y2 - min_y2)

    # get length and width of intersection
    intersection_x_length = min(max_x1, max_x2) - max(min_x1, min_x2)
    intersection_y_length = min(max_y1, max_y2) - max(min_y1, min_y2)

    # IoU
    if intersection_x_length > 0 and intersection_y_length > 0:
        intersection_area = intersection_x_length * intersection_y_length
        union_area = rect1_area + rect2_area - intersection_area
        ret = intersection_area / union_area
    return ret
