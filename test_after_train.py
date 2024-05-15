import torchvision
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import os
from utility import visualize_predictions, read_image, remove_low_confidence_predictions
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from map50 import calculate_mAP50
import json
from torchmetrics.detection.mean_ap import MeanAveragePrecision

CONF_THRESH = 0.4

if __name__ == '__main__':
    model = torch.load('models/entire_model.pth')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    model.eval()  # Set the model to evaluation mode
    image_path = 'split_data/valid/images/13.jpg'
    label_path = 'split_data/valid/labels_for_faster-rcnn/13.json'

    res = model([read_image(image_path).to(device)])
    with open(label_path) as f:
        ground_truth = dict(json.load(f))
    ground_truth['boxes'] = torch.Tensor(ground_truth['boxes'])
    ground_truth['labels'] = torch.Tensor(ground_truth['labels'])
    res = [remove_low_confidence_predictions(pred, CONF_THRESH) for pred in res]
    print(f"mAP50: {calculate_mAP50(res, [ground_truth])}")
    visualize_predictions(Image.open(image_path).convert('RGB'), res[0])
    print(res)

    
