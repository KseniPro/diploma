import torchvision
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import os
from utility import visualize_predictions, read_image, read_label, remove_low_confidence_predictions
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
    
    image_dir = 'split_data/valid/images'
    label_dir = 'split_data/valid/labels_for_faster-rcnn'

    images_paths = sorted([os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir)])[:10]
    label_paths =  sorted([os.path.join(label_dir, label_name) for label_name in os.listdir(label_dir)])[:10]

    images = [read_image(image_file).to(device) for image_file in images_paths]
    labels = [read_label(label_file) for label_file in label_paths]

    with torch.no_grad():
        res = model(images)
    res = [remove_low_confidence_predictions(pred, CONF_THRESH) for pred in res]
    print(f"mAP50: {calculate_mAP50(res, labels)}")
    #visualize_predictions(Image.open(image_path).convert('RGB'), res[0])
    #print(res)

    
