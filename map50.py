import numpy as np

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    
    # Calculate the (x, y)-coordinates of the intersection rectangle
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    
    # Calculate the area of intersection rectangle
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    
    # Calculate the area of both the prediction and ground-truth rectangles
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    
    # Calculate the union area
    union_area = box1_area + box2_area - inter_area
    
    # Calculate the IoU
    iou = inter_area / union_area
    
    return iou

def calculate_map50(predictions, ground_truths):
    """
    Calculate mAP@0.5 given predictions and ground truths.
    predictions and ground_truths are dictionaries with 'boxes', 'labels', and 'scores'.
    """
    # Flatten all boxes, labels, and scores
    pred_boxes = predictions['boxes'].cpu().detach().numpy()
    true_boxes = ground_truths['boxes']
    scores = predictions['scores'].cpu().detach().numpy()
    
    # Sort predictions by scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    pred_boxes = pred_boxes[sorted_indices]
    
    # Initialize variables to keep track of matches
    true_positives = np.zeros(len(pred_boxes))
    false_positives = np.zeros(len(pred_boxes))
    
    # Track which ground truth boxes have been matched
    matched = np.zeros(len(true_boxes))
    
    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        for j, true_box in enumerate(true_boxes):
            iou = calculate_iou(pred_box, true_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        # If the best match has an IoU > 0.5 and is not matched already
        if best_iou > 0.5 and matched[best_gt_idx] == 0:
            true_positives[i] = 1  # It's a match
            matched[best_gt_idx] = 1  # Mark this ground truth box as matched
        else:
            false_positives[i] = 1  # No match, or already matched
    
    # Calculate precision and recall arrays
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)
    recalls = tp_cumsum / len(true_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    print(true_positives, false_positives)
    # Calculate AP for this class
    ap = np.trapz(precisions, recalls)
    
    return ap
