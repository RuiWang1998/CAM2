from pprint import pprint as pp

import numpy as np
import torch

from YOLOSensitivity import YOLOMeasurer
from modified_YOLO import YOLO

config_path = 'YOLOv3/config/yolov3.cfg'
weight_path = "YOLOv3/weights/yolov3.weights"
image_folder = "YOLOv3/data/samples"
class_path = 'YOLOv3/data/coco.names'
image_size = 416
YOLOv3 = YOLO(config_path, image_size)
YOLOv3.load_weights(weight_path)
yolo_module_list = list(YOLOv3.children())[0]
device = "cpu"  # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
inputs = torch.tensor(np.loadtxt("input.csv", np.float)).view(1, 3, 416, 416).type(torch.float)
inputs.requires_grad = True
measurer = YOLOMeasurer(YOLOv3, yolo_module_list, cuda=False)
measurer.model.eval()

layer_idx = [i for i in range(measurer.layer_num)]
all_outputs = measurer.model(inputs, layer_idx=layer_idx)

outputs = all_outputs[0]

conv_weight = measurer.model.module_list[0][0].weight.type(torch.float)
bn = measurer.model.module_list[0][1]
bn_weight = bn.weight / torch.sqrt(bn.running_var - bn.eps)

grads_manual = torch.zeros_like(inputs)
padder = torch.nn.ConstantPad2d(1, 0)
padded = padder(grads_manual.clone()).type(torch.float)

leaky_output_gradient = (outputs > 0).type(torch.float) * 0.9 + 0.1
weights = torch.stack([torch.stack([conv_weight for _ in range(image_size)], 1) for _ in range(image_size)], 1)
weights = torch.unsqueeze(weights, 0)
bned_leaky = torch.einsum("ijkl, j->ijkl", leaky_output_gradient, bn_weight)
bned_leaky_weight = torch.einsum("ijkl, ijklmno->ijklmno", bned_leaky, weights.type(torch.float))

leaky_mean_weight = bned_leaky_weight.mean(1).permute(0, 3, 1, 2, 4, 5)

for i in range(3):
    for j in range(3):
        padded[:, :, i:image_size + i, j:image_size + j] += leaky_mean_weight[..., i, j]

grads_manual = padded[:, :, 1:-1, 1:-1] / image_size / image_size

pp(grads_manual.mean())
