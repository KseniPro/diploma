import torchvision
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import os
from dataset import StampDataset
from utility import collate_fn, train_one_epoch, visualize_predictions, read_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

if __name__ == '__main__':
    model = torch.load('models/entire_model.pth')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    model.eval()  # Set the model to evaluation mode
    image_path = 'split_data/valid/images/13.jpg'
    res = model([read_image(image_path).to(device)])
    print(res)
    visualize_predictions(Image.open(image_path).convert('RGB'), res[0])

    
