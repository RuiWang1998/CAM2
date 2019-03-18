import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from YOLOv3.models import Darknet
from pprint import pprint as pp
from YOLOv3.utils.datasets import ImageFolder
from YOLOv3.utils.utils import load_classes
import os

device = 'cpu'# torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
image_size = 416
config_path = 'YOLOv3/config/yolov3.cfg'
weight_path = "YOLOv3/weights/yolov3.weights"
image_folder = "YOLOv3/data/samples"
class_path = 'YOLOv3/data/coco.names'
batch_size = 1
n_cpu = 8

model = Darknet(config_path, image_size)
model.load_weights(weight_path)
model.to(device)

params = model.state_dict()

dataloader = DataLoader(ImageFolder(image_folder, img_size=image_size),
                            batch_size=batch_size, shuffle=False, num_workers=n_cpu)
classes = load_classes(class_path)

imgs = []
img_detections = []
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    imgs.append(input_imgs)

out = model(imgs[0], layer_idx={0, 1, 2, 10, 100})

plt.imshow(out[100][:, 100, :, :].squeeze().detach())
plt.show()

