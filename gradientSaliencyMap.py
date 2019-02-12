#%% [markdown]
# # Saliency Map
# This is my implementation of saliency map that figures out how much each of the 
# pixels matters in the process of YOLOv3

#%%
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

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
                        batch_size=batch_size, shuffle=False, num_workers=n_cpu)

classes = load_classes(class_path)
#%%
imgs = []
img_detections = []

#%%
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs.requires_grad = True
    imgs.append(input_imgs)

#%% [markdown]
# # A simple demonstration
# Here the result output would be of 7 dimensions. The first four are for the coordinates of the bounding boxes' angles, 
# the fifth is the confidence that there is an object, the sixth is the confidence that the object belongs to that class,
# which is the last output

#%%
input_imgs = imgs[0].to(device)
input_imgs = Variable(input_imgs.data, requires_grad=True)
output = model(input_imgs)

#%% [markdown]
# ## The output of the model is of shape (batch_size, num_boxes, num_classes + 5)
# ###
# #### For each of the bounding boxes, the first four is the spatial information of the box
# #### and the fifth element is the confidence of this bounding box. THe rest of the output
# #### should be the probability of each of the class that this box is predicting

#%% [markdown]
# ### this code below gives which of the boxes have a confidence level above 0.9
confidence_mask = output[:, :, 4] > 0.9
confidence_index = confidence_mask.nonzero()
confidence_index = [entry[1] for entry in confidence_index]
confidence_preds, class_preds = torch.max(output[confidence_mask][:,5:85], dim=1)
conf_class = {conf:classes[class_preds[i]] for i,conf in enumerate(confidence_preds)}
class_conf = {class_id:[] for class_id in conf_class.values()}
for conf, class_item in conf_class.items():
    class_conf[class_item].append(conf)
class_conf = {class_id:max(value) for class_id, value in class_conf.items()}

#%%
confidences = output[:, :, 4][0][confidence_index[0]]

#%% [markdown]
# ### Let's first try to find out what the first box predicts
class_conf, class_pred = torch.max(output[:, confidence_index[0], 5:85], dim=1)
classes[class_pred]

#%%
class_conf.backward()

#%%


#%%
saliency = input_imgs.grad[0].cpu().permute(1,2,0)

#%% [markdown]
# # The Saliency Map
# ### Below are some mathematical operations that help visualization better
saliency_up = torch.abs(saliency) * 1e7
saliency_normed = (saliency_up - saliency_up.min())/(saliency_up.max() - saliency_up.min())
saliency_normed_scaled = (saliency_normed + 0.7)**2
saliency_scaled_nonlinear = torch.log(saliency_normed_scaled)
saliency_pic = (saliency_scaled_nonlinear - saliency_scaled_nonlinear.min())/(saliency_scaled_nonlinear.max() - saliency_scaled_nonlinear.min())
plt.imshow(saliency_pic)

#%% [markdown]
# The actual image
plt.imshow(input_imgs[0].permute(1,2,0).detach())

#%% [markdown]
# # Let's organize and set this into a function
def img2Saliency(input_imgs, threshold=0.9):
    input_imgs = input_imgs.to(device)
    input_imgs = Variable(input_imgs.data, requires_grad=True)
    output = model(input_imgs)

    confidence_mask = output[:, :, 4] > threshold

    confidence_preds, class_preds = torch.max(output[confidence_mask][:,5:85], dim=1)
    conf_class = {conf:classes[class_preds[i]] for i,conf in enumerate(confidence_preds)}
    class_conf = {class_id:[] for class_id in conf_class.values()}
    for conf, class_item in conf_class.items():
        class_conf[class_item].append(conf)
    class_conf = {class_id:max(value) for class_id, value in class_conf.items()}

    saliency_maps = []
    for i, name in enumerate(class_conf.keys()):
        output2 = model(input_imgs)
        confidence_mask2 = output2[:, :, 4] > threshold
        confidence_preds2, class_preds2 = torch.max(output2[confidence_mask2][:,5:85], dim=1)
        conf_class2 = {conf:classes[class_preds2[i]] for i,conf in enumerate(confidence_preds2)}
        class_conf2 = {class_id:[] for class_id in conf_class2.values()}
        for conf, class_item in conf_class2.items():
            class_conf2[class_item].append(conf)
        class_conf2 = {class_id:max(value) for class_id, value in class_conf2.items()}
        class_conf2[name].backward()
        print(name)
        
        saliency = input_imgs.grad[0].cpu().permute(1,2,0)

        saliency_up = torch.abs(saliency) * 1e7
        saliency_normed = (saliency_up - saliency_up.min())/(saliency_up.max() - saliency_up.min())
        saliency_normed_scaled = (saliency_normed + 0.7)**2
        saliency_scaled_nonlinear = torch.log(saliency_normed_scaled)
        saliency_pic = (saliency_scaled_nonlinear - saliency_scaled_nonlinear.min())/(saliency_scaled_nonlinear.max() - saliency_scaled_nonlinear.min())
        saliency_maps.append(saliency_pic)

    columns = 1
    rows = len(class_conf) + 1
    fig=plt.figure(figsize=(8, 8))
    for i in range(1, columns*rows):
        img = saliency_maps[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)

    fig.add_subplot(rows, columns, rows)
    plt.imshow(input_imgs[0].permute(1,2,0).detach())
    plt.show()


#%%
img2Saliency(imgs[0])

#%%
plt.imshow(imgs[1][0].permute(1,2,0).detach().numpy())

#%%
