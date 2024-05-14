import torchvision
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import os
from dataset import StampDataset
from utility import collate_fn, train_one_epoch, visualize_predictions, read_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from map50 import calculate_map50
import json

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

    print(f"mAP50: {calculate_map50(res[0], ground_truth)}")
    visualize_predictions(Image.open(image_path).convert('RGB'), res[0])

    
