import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


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