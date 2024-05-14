import torchvision
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import os
from dataset import StampDataset
from utility import collate_fn, train_one_epoch, visualize_predictions
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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

    val_dir = os.path.join('split_data', 'valid')
    val_dataset = StampDataset(val_dir)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn = collate_fn
    )
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    num_classes = 3
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    num_epochs = 5

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
    torch.save(model, 'models/entire_model.pth')

    # model.eval()
    # res = model(val_data_loader)
    # visualize_predictions(res)
    
