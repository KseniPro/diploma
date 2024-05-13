import torchvision
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import cv2
import os
from dataset import StampDataset
from utility import collate_fn

if __name__ == '__main__':
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # For training
    train_dir = os.path.join('split_data', 'train')
    train_dataset = StampDataset(train_dir)
    
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn = collate_fn
    )
    
    it = iter(train_data_loader)
    first = next(it)
    print(first)