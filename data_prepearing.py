import os
import json
import cv2
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import torchvision

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torch.nn.utils.rnn import pad_sequence

from model_training.utils import StampDataset


def collate_fn(batch):
    imgs, annotations = zip(*batch)
    imgs = torch.stack(imgs)
    annotations = pad_sequence([torch.tensor(a) for a in annotations], batch_first=True, padding_value=0)
    return imgs, annotations

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = StampDataset(root_dir='code/split_data/train', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)
valid_dataset = StampDataset(root_dir='code/split_data/valid', transform=None)
valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)
print(len(train_dataloader), len(valid_dataloader))

os.environ['OMP_NUM_THREADS'] = '1'


def show_batch_simple(example_batch, n=4):
    concatenated = example_batch[0][:n]
    grid_img = torchvision.utils.make_grid(concatenated, nrow=n)

    fig, ax = plt.subplots(figsize=(36, 4))
    ax.imshow(grid_img.permute(1, 2, 0), interpolation='bilinear')
    print(example_batch[1][:n].numpy())
    
example_batch = next(iter(train_dataloader))
show_batch_simple(example_batch, n=4)