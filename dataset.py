
import os
from torch.utils.data import Dataset
from utility import read_boxes_and_labels, read_image
import torch

class StampDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.images = sorted([f'{root_dir}/images/{filename}' for filename in (os.listdir(f'{root_dir}/images/'))])
        self.labels = sorted([f'{root_dir}/labels_for_faster-rcnn/{filename}' for filename in (os.listdir(f'{root_dir}/labels_for_faster-rcnn/'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_path = self.labels[idx]

        image = read_image(image_path)
        boxes, labels = read_boxes_and_labels(label_path)
        if self.transforms:
            image = self.transforms(image)
        target = {
            'boxes': boxes, 
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros((boxes.shape[0]), dtype=torch.int64)
        }

        return image, target