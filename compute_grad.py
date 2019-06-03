import torch
import torch.nn as nn


def grad_compute_seq(outputs, input_shape, module, negative_slope=0.1, reduction='mean'):
    """
    This function calculates the gradient of the sequential model like:
        Conv2d->BatchNorm2d->LeakyReLU
    :param outputs: the output to calculate the gradient for
    :param input_shape: the shape of the input to the model
    :param module: of the form Conv2d->BatchNorm2d->LeakyReLU
    :param reduction: the reduction method
    :return: the gradient
    """
    conv = module[0]
    padding = conv.padding
    padder = nn.ConstantPad2d(padding[0], 0)
    kernel_size = conv.kernel_size
    stride = conv.stride

    leaky_output_gradient = (outputs > 0).type(torch.float) * (1 - negative_slope) + negative_slope
    conv_weight = conv.weight
    weights = torch.stack(
        [torch.stack([conv_weight for _ in range(outputs.shape[2])], 1) for _ in range(outputs.shape[3])], 1)
    weights = torch.unsqueeze(weights, 0)

    bn = module[1]
    bn_weight = bn.weight / torch.sqrt(bn.running_var - bn.eps)
    bned_leaky = torch.einsum("ijkl, j->ijkl", leaky_output_gradient, bn_weight)
    bned_leaky_weight = torch.einsum("ijkl, ijklmno->ijklmno", bned_leaky, weights.type(torch.float))
    leaky_mean_weight = bned_leaky_weight.sum(1).permute(0, 3, 1, 2, 4, 5)

    padded = padder(torch.zeros([1, conv.in_channels, input_shape[0], input_shape[1]])).type(torch.float)

    padded.requires_grad = False

    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            x_indices = range(i * stride[0], min(input_shape[0] + i * stride[0],
                                                 input_shape[0] + padding[0] * 2 - 1), stride[0])
            y_indices = range(j * stride[1], min(input_shape[1] + j * stride[1],
                                                 input_shape[1] + padding[1] * 2 - 1), stride[1])
            padded[:, :, x_indices, :][:, :, :, y_indices] \
                += leaky_mean_weight[..., i, j]

    if reduction == 'mean':
        return padded[:, :, padding[0]:-padding[0], padding[1]:-padding[1]] \
               / outputs.shape[1] / input_shape[0] / input_shape[1]
    else:
        return padded[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]


if __name__ == "__main__":
    import numpy as np
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
    inputs.requires_grad = False
    measurer = YOLOMeasurer(YOLOv3, yolo_module_list, cuda=False)
    measurer.model.eval()
    layer_idx = [i for i in range(measurer.layer_num)]
    grad = measurer.model(inputs, input_index=1, output_index=2)

    print(grad.mean())

    layer_idx = [i for i in range(measurer.layer_num)]
    all_outputs = measurer.model(inputs, layer_idx=layer_idx)

    outputs = all_outputs[1]
    manual = grad_compute_seq(outputs, (416, 416), measurer.model.module_list[1], negative_slope=0.1, reduction='mean')
    print(manual.mean())
