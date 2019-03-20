#%%
import torch
import torch.nn as nn
import torch.optim as optim

#%%
device = 'cpu' # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#%%
from YOLOv3.models import Darknet
image_size = 416
config_path = 'YOLOv3/config/yolov3.cfg'
weight_path = "YOLOv3/weights/yolov3.weights"
image_folder = "YOLOv3/data/samples"
class_path = 'YOLOv3/data/coco.names'
batch_size = 1
EPOCHS = 10
BATCH_SIZE = 1

#%% [markdown]
# # Select the model

#%%
model = Darknet(config_path, image_size)
model.load_weights(weight_path)
model.to(device)

#%% [markdown]
# ## Miscellaneous functions 


#%%
def tanh(x, scaling=0.2):
    x = torch.nn.Tanh()(scaling * x)
    return x / 2 + 0.5

#%% [markdown]
# # This set the image sizes in the intermmediate steps


#%%
size_seq = [40, 120, 360, image_size]

#%% [markdown]
# ## This deals with upscaling

#%%
import scipy.misc as im


def upscale(orig_img, out_idx):
    out_res = size_seq[out_idx] / len(orig_img)
    out_img = im.imresize(orig_img, out_res)
    out_img /= 255
    return out_img


def upscale_torch(orig_img):
    out_res = image_size / len(orig_img)
    out_img = im.imresize(orig_img, out_res)
    out_img /= 255
    return torch.tensor(out_img).to(device)


#%%
z = torch.randn([BATCH_SIZE, 3, image_size, image_size]).to(device)
z.requires_grad = True
optimizer = optim.Adam([z], lr=1e-2, weight_decay=1e-3)


#%%
layer_idx = 101
channel_idx = 3
