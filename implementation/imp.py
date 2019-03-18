import torch
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.optim import Adam as adam
import scipy.misc

device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
vgg = models.vgg19(pretrained=True).to(device)
vgg.eval()
module_list = [layer for layer in vgg.children()]
BATCH_SIZE = 1
IMAGE_SIZE = 224


def get_act(inputs, layer_idx, module_list=module_list):
    for i, layer in enumerate(module_list):
        if i == 0:
            outputs = layer(inputs)
        else:
            outputs = layer(outputs)
        if i == layer_idx:
            return outputs


def tanh(x, scaling=0.2):
    x = torch.nn.Tanh()(scaling * x)
    return x / 2 + 0.5


layer_idx = 0
channel_idx = 0

z = torch.randn([BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE]).to(device)
optimizer = adam([z], lr=1e-1, weight_decay=1e-3)

EPOCHS = 10000

for i in range(EPOCHS):
  optimizer.zero_grad()
  activation = get_act(tanh(z), layer_idx)
  channel_act = - activation[0, channel_idx].mean(-1).mean(-1)
  channel_act.backward()
  optimizer.step()
  if i % 30 == 0:
      image = tanh(z[0]).detach().cpu().permute(1, 2, 0)
      scipy.misc.imsave(f"cimage{i}.jpg", image)
      scipy.misc.imsave(f'c1image{i}.jpg', image[:, :, 0])
      scipy.misc.imsave(f'c2image{i}.jpg', image[:, :, 1])
      scipy.misc.imsave(f'c3image{i}.jpg', image[:, :, 2])
      print(f"Epoch {i}, loss {channel_act}")
