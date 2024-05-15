import cv2
import numpy as np
import json
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import math
import sys
import utils


def read_image(path):
    image = Image.open(path).convert('RGB')

    transform = T.Compose([
        T.ToTensor(),
        #T.Resize(target_size, antialias=True)
    ])
    
    image_tensor = transform(image)

    return image_tensor

def read_boxes_and_labels(path):
    with open(path, 'r') as file:
        data = json.load(file)
    boxes = data["boxes"]
    labels = data["labels"]
    return torch.as_tensor(boxes, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.int64)

def visualize_predictions(image, predictions):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    for i in range(predictions["boxes"].shape[0]):
        box = predictions["boxes"][i]
        label = predictions["labels"][i]
        score = predictions["scores"][i]

        if score < 0.1:
            continue

        x, y, xmax, ymax = box
        rect = patches.Rectangle((x, y), xmax - x, ymax - y, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, f'{label}: {score:.2f}', fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.show()

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def remove_low_confidence_predictions(preds, thresh):
    to_remove = []
    for i, score in enumerate(preds['scores']):
        if score < thresh:
            to_remove.append(i)

    def remove_elements_at_indices(elements, indices):
        return [element for i, element in enumerate(elements) if i not in indices]
    
    preds['scores'] = np.array(remove_elements_at_indices(preds['scores'].cpu().detach().numpy(), to_remove))
    preds['boxes'] = np.array(remove_elements_at_indices(preds['boxes'].cpu().detach().numpy(), to_remove))
    preds['labels'] = np.array(remove_elements_at_indices(preds['labels'].cpu().detach().numpy(), to_remove))

    return preds

def read_label(label_path):
    with open(label_path) as f:
        ground_truth = dict(json.load(f))

    ground_truth['boxes'] = torch.Tensor(ground_truth['boxes'])
    ground_truth['labels'] = torch.Tensor(ground_truth['labels'])
    
    return ground_truth