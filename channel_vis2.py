import torch
import matplotlib.pyplot as plt
from YOLOv3.models import Darknet
import scipy.misc
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_size = 416
config_path = 'YOLOv3/config/yolov3.cfg'
weight_path = "YOLOv3/weights/yolov3.weights"
image_folder = "YOLOv3/data/samples"
class_path = 'YOLOv3/data/coco.names'
batch_size = 1
EPOCHS = 500

model = Darknet(config_path, image_size)
model.load_weights(weight_path)
model.to(device)
model.eval()


def tanh(x, scaling=0.2):
    x = torch.nn.Tanh()(scaling * x)
    return x / 2 + 0.5


z = torch.randn([batch_size, 3, image_size, image_size]).to(device)
z.requires_grad = True
optimizer = torch.optim.Adam([z], lr=1e-1, weight_decay=1e-6)

layer_idx = 101
channel_idx = 12

for i in range(EPOCHS):

    optimizer.zero_grad()
    layer_act = model(tanh(z), layer_idx={layer_idx})
    # channel_act = - layer_act[layer_idx][0, channel_idx, 20, 20]
    layer_act_sum = - layer_act[layer_idx].mean(-1).mean(-1).mean(0)
    channel_act = layer_act_sum[channel_idx]

    channel_act.backward()
    optimizer.step()
    if i % 500 == 0:
        image = tanh(z[0]).detach().cpu().permute(1, 2, 0)
        scipy.misc.imsave(f"cimage{i}.jpg", image)
        scipy.misc.imsave(f'c1image{i}.jpg', image[:, :, 0])
        scipy.misc.imsave(f'c2image{i}.jpg', image[:, :, 1])
        scipy.misc.imsave(f'c3image{i}.jpg', image[:, :, 2])
    print(f"Epoch {i}")
