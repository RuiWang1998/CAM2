#%% [markdown]
# # Saliency Map
# This is my implementation of saliency map that figures out how much each of the 
# pixels matters in the process of YOLOv3

#%%
import torch
import torchvision
from torch.utils.data import DataLoader

#%% [markdown]
# # constants
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
image_size = 416
config_path = 'YOLOmodels/config/yolov3.cfg'
weight_path = "C:/Users/ruiwa/Documents/GitHub/PyTorch-YOLOv3/weightsyolov3.weights"
image_folder = 'samples'
class_path = 'YOLOmodels/classes/coco.names'
batch_size = 1
n_cpu = 8

#%% [markdown]
# # the Model
#%%
from YOLOmodels.models import Darknet
model = Darknet(config_path, image_size)
model.load_weights(weight_path)
model.to(device)

#%%
from YOLOmodels.utils.datasets import *
from YOLOmodels.utils.utils import *
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
    output = model(input_imgs)


#%%
