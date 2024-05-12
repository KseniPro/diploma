import os
import json
from PIL import Image


images_dir = '/Users/admin/Documents/ВКР/code/split_data/train/images'
labels_dir = '/Users/admin/Documents/ВКР/code/split_data/train/labels'
output_dir = '/Users/admin/Documents/ВКР/code/split_data/train/labels_for_faster-rcnn'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(images_dir):
    image_path = os.path.join(images_dir, filename)
    img = Image.open(image_path)
    image_width, image_height = img.size
    
    basename = os.path.splitext(filename)[0]
    yolo_path = os.path.join(labels_dir, basename + '.txt')
    with open(yolo_path, 'r') as f:
        lines = f.readlines()
        
    boxes = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        x1 = (x_center - width / 2) * image_width
        y1 = (y_center - height / 2) * image_height
        x2 = (x_center + width / 2) * image_width
        y2 = (y_center + height / 2) * image_height
        boxes.append([x1, y1, x2, y2])
        
    output_path = os.path.join(output_dir, basename + '.json')
    with open(output_path, 'w') as f:
        json.dump({'boxes': (boxes), 'labels': [0]*len(boxes)}, f)