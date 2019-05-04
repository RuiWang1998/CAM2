import gc

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
place_holder = torch.randn(1, 3, image_size, image_size).to(device)
measurer = YOLOMeasurer(YOLOv3, yolo_module_list, cuda=False)

layer_idx = [i for i in range(measurer.layer_num)]
all_outputs = measurer.model(place_holder, layer_idx=layer_idx)
grads = measurer.model(place_holder, input_index=104, output_index=105)
# So we need to take the derivative from layer 104's output to layer 103 output
# layer_103_output = all_outputs[103]
# layer_104_output = all_outputs[104]
# activation_grad = (layer_104_output > 0).type(torch.float64) * 0.9 + 0.1
# filter_weights = measurer.module_list[104][0].weight

grads = measurer.model(place_holder, input_index=0, output_index=1)
activation_grad = (all_outputs[0] > 0).type(torch.float64) * 0.9 + 0.1
filter_weights = measurer.module_list[0][0].weight

grads_manual = torch.zeros_like(place_holder)
layer_output = all_outputs[0]
leaky_output_gradient = (layer_output > 0).type(torch.float) * 0.9 + 0.1


def in_limit(value, limit):
    return limit[0] <= value < limit[1]


one_channel = leaky_output_gradient[0][0]
for i in range(416):
    for j in range(416):
        image_size_limit = (0, 416)
        for ii in range(-1, 2):
            for ji in range(-1, 2):
                if in_limit(i + ii, image_size_limit) and in_limit(j + ji, image_size_limit):
                    input_index = (i + ii, j + ji)
                    grads_manual[0, :, input_index[0], input_index[1]] += \
                        torch.matmul(leaky_output_gradient[0, :, input_index[0], input_index[1]],
                                     filter_weights[:, :, ii, ji])
                    gc.collect()
