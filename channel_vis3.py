import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from YOLOv3.models import Darknet
import scipy.misc
import numpy as np
from models_vis import Rectifier
import torchvision
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_size = 416
config_path = 'YOLOv3/config/yolov3.cfg'
weight_path = "YOLOv3/weights/yolov3.weights"
image_folder = "YOLOv3/data/samples"
class_path = 'YOLOv3/data/coco.names'
BATCH_SIZE = 1
EPOCHS = 3

transf = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),])

dataset = torchvision.datasets.ImageFolder(root=('COCO/'),
                                                     transform=transf)
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

model = Darknet(config_path, image_size)
model.load_weights(weight_path)
model.to(device)

rectifier = Rectifier().to(device)
rectifier.to(device)

rectifier.train()
model.eval()


def tanh(x, scaling=0.2):
    x = torch.nn.Tanh()(scaling * x)
    return x / 2 + 0.5


z = torch.randn([BATCH_SIZE, 3, image_size, image_size]).to(device)
z.requires_grad = True
optimizer1 = torch.optim.Adam([z], lr=1e-2, weight_decay=1e-3)
optimizer2 = torch.optim.Adam(rectifier.parameters())
loss_fn = nn.BCELoss()

layer_idx = 101
channel_idx = 3
zeros = torch.zeros([BATCH_SIZE]).to(device)
ones = torch.ones([BATCH_SIZE]).to(device)

for i in range(EPOCHS):

    for j, (image, label) in enumerate(data_loader):

        optimizer2.zero_grad()
        validity = rectifier(tanh(z))
        loss_fake = loss_fn(validity, zeros)
        image = image.to(device)
        real_val = rectifier(image)
        loss_real = loss_fn(real_val, ones)
        loss = loss_fake + loss_real
        loss.backward()
        optimizer2.step()

        optimizer1.zero_grad()
        layer_act = model(tanh(z), layer_idx={layer_idx})
        # channel_act = - layer_act[layer_idx][0, channel_idx, 20, 20]
        layer_act_sum = - layer_act[layer_idx].mean(-1).mean(-1).mean(0)
        channel_act = layer_act_sum[channel_idx]

        validity = rectifier(tanh(z))
        loss_fake = loss_fn(validity, ones)
        loss = validity + channel_act
        loss.backward()
        optimizer1.step()

        if j % 500 == 0:
            image = tanh(z[0]).detach().cpu().permute(1, 2, 0)
            scipy.misc.imsave(f"{i}cimage{j}.jpg", image)
            scipy.misc.imsave(f'{i}c1image{j}.jpg', image[:, :, 0])
            scipy.misc.imsave(f'{i}c2image{j}.jpg', image[:, :, 1])
            scipy.misc.imsave(f'{i}c3image{j}.jpg', image[:, :, 2])
            print(f"Epoch {i}, image {j}")
