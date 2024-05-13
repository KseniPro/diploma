import torchvision
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import cv2
import os
from utility import read_image
from torchvision.transforms import functional as F
from utility import visualize_predictions
from PIL import Image

if __name__ == '__main__':
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    image = read_image('image.png')
    model.eval()
    print("PREDICTING")
    predictions = model([image])
    visualize_predictions(Image.open('image.png').convert('RGB'), predictions[0])