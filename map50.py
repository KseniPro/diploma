import numpy as np

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area

    return iou

def calculate_mAP50(predictions, ground_truths):
    global_data = []
    total_true_boxes = 0
    for prediction, gt in zip(predictions, ground_truths):
        pred_labels = prediction['labels'].cpu().detach().numpy()
        pred_boxes = prediction['boxes'].cpu().detach().numpy()
        scores = prediction['scores'].cpu().detach().numpy()
        true_boxes = gt['boxes']
        total_true_boxes += sum(pred_labels == 1)
        matched = np.zeros(len(true_boxes))
        for i, pred_box in enumerate(pred_boxes):
            if (pred_labels[i] != 1):
                continue

            best_iou = 0
            best_gt_idx = -1
            for j, true_box in enumerate(true_boxes):
                iou = calculate_iou(pred_box, true_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            is_true_positive = 0
            if best_iou > 0.5 and matched[best_gt_idx] == 0:
                is_true_positive = 1 # It's a match
                matched[best_gt_idx] = 1  # Mark this ground truth box as matched

            global_data.append((scores[i], is_true_positive))

    print(sorted(global_data, reverse=True, key = lambda tup: tup[0]))
    true_positives = np.array([tup[1] for tup in global_data])
    false_positives = (true_positives == 0).astype(np.uint32)
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)
    recalls = tp_cumsum / len(true_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    # Calculate AP for this class
    ap = np.trapz(precisions, recalls)

    return ap
            
        