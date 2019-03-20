import os

import scipy.misc
import torch

from YOLOv3.models import Darknet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_size = 416
config_path = 'YOLOv3/config/yolov3.cfg'
weight_path = "YOLOv3/weights/yolov3.weights"
image_folder = "YOLOv3/data/samples"
class_path = 'YOLOv3/data/coco.names'
batch_size = 1
EPOCHS = 10

model = Darknet(config_path, image_size)
model.load_weights(weight_path)
model.to(device)


def tanh(x, scaling=0.2):
    x = torch.nn.Tanh()(scaling * x)
    return x / 2 + 0.5

model.eval()
for layer_idx in range(106):
    try:
        path = f"visuals/layer{layer_idx}"
        os.mkdir(path)
        path = f"visuals/layer{layer_idx}/Color"
        os.mkdir(path)
        path = f"visuals/layer{layer_idx}/Mono0"
        os.mkdir(path)
        path = f"visuals/layer{layer_idx}/Mono1"
        os.mkdir(path)
        path = f"visuals/layer{layer_idx}/Mono2"
        os.mkdir(path)
    except OSError:
        pass
    for channel_idx in range(1024):
        try:
            z = torch.rand([batch_size, 3, image_size, image_size]).to(device)
            z.requires_grad = True
            optimizer = torch.optim.Adam([z], lr=1e0, weight_decay=1e-1)
            for i in range(EPOCHS):
                optimizer.zero_grad()
                layer_act = model(tanh(z), layer_idx={layer_idx})
                # layer_act_sum = layer_act[layer_idx][0, :, 0, 0]
                layer_act_sum = - layer_act[layer_idx].mean(-1).mean(-1).mean(0)
                channel_act = layer_act_sum[channel_idx]

                channel_act.backward()
                optimizer.step()
                if (i + 1) % EPOCHS == 0:
                    scipy.misc.imsave(f'visuals/layer{layer_idx}/Color/layer{layer_idx}_channel{channel_idx}.jpg',
                                      tanh(z[0]).detach().cpu().permute(1, 2, 0)[:, :, :])
                    scipy.misc.imsave(f'visuals/layer{layer_idx}/Mono0/layer{layer_idx}_channel{channel_idx}.jpg',
                                      tanh(z[0]).detach().cpu().permute(1, 2, 0)[:, :, 0])
                    scipy.misc.imsave(f'visuals/layer{layer_idx}/Mono1/layer{layer_idx}_channel{channel_idx}.jpg',
                                      tanh(z[0]).detach().cpu().permute(1, 2, 0)[:, :, 1])
                    scipy.misc.imsave(f'visuals/layer{layer_idx}/Mono2/layer{layer_idx}_channel{channel_idx}.jpg',
                                      tanh(z[0]).detach().cpu().permute(1, 2, 0)[:, :, 2])
            print(f"Layer {layer_idx} | channel {channel_idx}")
        except IndexError:
            pass
