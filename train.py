
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import os
from utils import StampDataset
from torch.utils.data import  DataLoader
from data_preparing import collate_fn, transform

if __name__ == '__main__':

    train_dataset = StampDataset(root_dir='code/split_data/train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)
    valid_dataset = StampDataset(root_dir='code/split_data/valid', transform=None)
    valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(len(train_dataloader), len(valid_dataloader))

    os.environ['OMP_NUM_THREADS'] = '1'

    num_classes = 1  # 1 класс (штамп) + фон
    model = fasterrcnn_resnet50_fpn(pretrained=True, weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10
    index = 0
    count = 0
    for epoch in range(num_epochs):
        for  images, labels in train_dataloader:
            loss_dict = model(images, labels)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            index += 1
            if index % 50 == 0:
                print(f'Iteration #{index} loss: {losses.item()}')
        print(f'Epoch #{epoch} loss: {losses.item()}')