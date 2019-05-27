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
place_holder1 = torch.randn(1, 3, image_size, image_size).to(device)
place_holder2 = place_holder1.detach().clone().to(device)
measurer = YOLOMeasurer(YOLOv3, yolo_module_list, cuda=False)
measurer.model.eval()

layer_idx = [i for i in range(measurer.layer_num)]
all_outputs = measurer.model(place_holder1, layer_idx=layer_idx)
# grads = measurer.model(place_holder, input_index=104, output_index=105)
# So we need to take the derivative from layer 104's output to layer 103 output
# layer_103_output = all_outputs[103]
# layer_104_output = all_outputs[104]
# activation_grad = (layer_104_output > 0).type(torch.float64) * 0.9 + 0.1
# filter_weights = measurer.module_list[104][0].weight

padder = torch.nn.ConstantPad2d(1, 0)
grads = measurer.model(place_holder2, input_index=0, output_index=1)
activation_grad = (all_outputs[0] > 0).type(torch.float64) * 0.9 + 0.1
filter_weights = measurer.module_list[0][0].weight  # this is the weights from the filter
batch_norm_weights = measurer.module_list[0][1].weight  # this is the weights for the

grads_manual = torch.zeros_like(place_holder1)
padded = padder(grads_manual.clone())
layer_output = all_outputs[0]
leaky_output_gradient = (layer_output > 0).type(torch.float) * 0.9 + 0.1
weights = torch.stack([torch.stack([filter_weights for _ in range(416)], 1) for _ in range(416)], 1)
weights = torch.unsqueeze(weights, 0)
bned_leaky = torch.einsum("ijkl, j->ijkl", leaky_output_gradient, batch_norm_weights)
bned_leaky_weight = torch.einsum("ijkl, ijklmno->ijklmno", bned_leaky, weights)
# leaky_weight = torch.einsum('ijklmno,ijkl->ijklmno', weights, leaky_output_gradient)
# bned_weight = torch.einsum("jklmnop,k->jklmnop", leaky_weight, batch_norm_weights)
leaky_mean_weight = bned_leaky_weight.mean(1).permute(0, 3, 1, 2, 4, 5)

for i in range(3):
    for j in range(3):
        padded[:, :, i:image_size + i, j:image_size + j] += leaky_mean_weight[..., i, j]
grads_manual = padded[:, :, 1:-1, 1:-1] / 416 / 416
grads_manual.mean()

(grads_manual == grads).sum()
grads.mean()

all_outputs = measurer.model(place_holder1, jacobian=True)
