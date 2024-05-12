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



def collate_fn(batch):
    imgs, annotations = zip(*batch)
    imgs = torch.stack(imgs)
    annotations = pad_sequence([torch.tensor(a) for a in annotations], batch_first=True, padding_value=0)
    return imgs, annotations





def show_batch_simple(example_batch, n=4):
    concatenated = example_batch[0][:n]
    grid_img = torchvision.utils.make_grid(concatenated, nrow=n)

    fig, ax = plt.subplots(figsize=(36, 4))
    ax.imshow(grid_img.permute(1, 2, 0), interpolation='bilinear')
    print(example_batch[1][:n].numpy())
