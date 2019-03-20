import torch

from visualizer import MultiStepVisualizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
image_size = 416
config_path = 'YOLOv3/config/yolov3.cfg'
weight_path = "YOLOv3/weights/yolov3.weights"
image_folder = "YOLOv3/data/samples"
class_path = 'YOLOv3/data/coco.names'
batch_size = 1
n_cpu = 8

from YOLOv3.models import Darknet

YOLOv3 = Darknet(config_path, image_size)
YOLOv3.load_weights(weight_path)


class YOLOv3Visualizer(MultiStepVisualizer):
    """
    This is a class that allows for some modifications for YOLOv3
    """

    def __init__(self, model):
        """
        First let us inherit the visualizer
        :param model: the YOLOv3 model pre-trained
        """
        super(YOLOv3Visualizer, self).__init__(model, device)

    def get_nth_output_layer(self, img, layer_idx):
        return self.model(img, layer_idx=layer_idx)
