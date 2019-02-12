#%% [markdown]
# # Saliency Map
# This is my implementation of saliency map that figures out how much each of the 
# pixels matters in the process of YOLOv3

#%%
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

#%% [markdown]
# # constants
device = 'cpu' #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
image_size = 416
config_path = 'C:/Users/ruiwa/Documents/GitHub/repos/ClonedModel/PyTorch-YOLOv3/config/yolov3.cfg'
weight_path = "C:/Users/ruiwa/Documents/GitHub/repos/ClonedModel/PyTorch-YOLOv3/weights/yolov3.weights"
image_folder = "C:/Users/ruiwa/Documents/GitHub/repos/ClonedModel/PyTorch-YOLOv3/data/samples"
class_path = 'C:/Users/ruiwa/Documents/GitHub/repos/ClonedModel/PyTorch-YOLOv3/data/coco.names'
batch_size = 1
n_cpu = 8

#%% [markdown]
# # the Model
#%%
# some_file.py
import sys
sys.path.insert(0, 'C:/Users/ruiwa/Documents/GitHub/repos/ClonedModel/PyTorch-YOLOv3')

from models import Darknet
model = Darknet(config_path, image_size)
model.load_weights(weight_path)
model.to(device)

#%%
from utils.datasets import *
from utils.utils import *
dataloader = DataLoader(ImageFolder(image_folder, img_size=image_size),
                        batch_size=batch_size, shuffle=True, num_workers=n_cpu)

classes = load_classes(class_path)
#%%
imgs = []
img_detections = []

#%%
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs.requires_grad = True

#%% [markdown]
# # A simple demonstration
# Here the result output would be of 7 dimensions. The first four are for the coordinates of the bounding boxes' angles, 
# the fifth is the confidence that there is an object, the sixth is the confidence that the object belongs to that class,
# which is the last output

#%%
input_imgs = input_imgs.to(device)
input_imgs = Variable(input_imgs.data, requires_grad=True)
output = model(input_imgs)

#%%
confidence_mask = output[:, :, 4] > 0.9
confidence_index = confidence_mask.nonzero()
confidence_index = [entry[1] for entry in confidence_index]
confidences = output[:, :, 4][0][confidence_index[0]]

#%%
confidences.backward()

#%%
import matplotlib.pyplot as plt

#%%
normalizer = nn.Sequential(
    nn.Tanh(),
    nn.Softmax()
)

#%%
saliency = input_imgs.grad[0].cpu().permute(1,2,0)

#%%
saliency_up = saliency * 1e7
# saliency_normed = normalizer(saliency_up)
saliency_normed = (saliency_up - saliency_up.min())/(saliency_up.max() - saliency_up.min()) - 0.3
plt.imshow(saliency_normed)

#%%
plt.imshow(input_imgs[0].detach().cpu().permute(1,2,0))

#%%
class_pred = output[:, :, 5:][0][confidence_index[0]]
class_pred.sort()

#%%
(class_pred == class_pred.max()).nonzero()

#%%
classes[7]

#%%


#%%


#%%


#%%
