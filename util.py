import os
import torch
import json

from PIL import Image
from torch.utils.data import Dataset

class StampDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f'{root_dir}/images/{filename}' for filename in (os.listdir(f'{root_dir}/images/'))]
        self.labels = [f'{root_dir}/labels/{filename}' for filename in (os.listdir(f'{root_dir}/labels/'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_path = self.labels[idx]
        with open(label_path, 'r') as f:
            data = json.load(f)  
            boxes = [list(map(float, box)) for box in data['boxes']]
            labels = list(map(int, data['labels']))
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        target = {'boxes': boxes, 'labels': labels}
        return image, target