import torch

from YOLOv3.models import Darknet


class YOLO(Darknet):
    """
    This is a modified version of YOLOv3 from https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/models.py
    This modification allows for more flexibility when it comes to investigation into the model
    """

    def __init__(self, config_path, img_size=416):
        super(YOLO, self).__init__(config_path, img_size)

    def forward(self, x, layer_idx=None, get_grad=None):
        """
        This function should allow for the following operations:
            1. get output from intermediate layers
            2. get intermediate gradient between layers
        :param x: the input
        :param layer_idx: the layer index for the above operations
        :param get_grad: whether to get gradient or not
        :return: depends
        """
        get_layer = layer_idx is not None
        activation_dict = dict()
        output = []
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                # Train phase: get loss
                x = module(x)
                output.append(x)
            layer_outputs.append(x)
            if get_layer:
                if i in layer_idx:
                    activation_dict[i] = x

        if get_layer:
            return activation_dict
        return torch.cat(output, 1)
