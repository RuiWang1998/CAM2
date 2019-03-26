import torch

from Sensitivity import SensitivityMeasurer
from YOLOv3.models import Darknet


class YOLOMeasurer(SensitivityMeasurer):
    """
    This is a measurer made for YOLO
    """

    def __init__(self, model, module_list, cuda=True):
        """
        First let's inherit the measurer
        :param model: the YOLOv3 model pre-trained
        :param module_list: the list of modules of the YOLO model
        :param cuda: the device to put on
        """
        super(YOLOMeasurer, self).__init__(model, module_list, cuda=cuda)


if __name__ == '__main__':
    config_path = 'YOLOv3/config/yolov3.cfg'
    weight_path = "YOLOv3/weights/yolov3.weights"
    image_folder = "YOLOv3/data/samples"
    class_path = 'YOLOv3/data/coco.names'

    image_size = 416
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    YOLOv3 = Darknet(config_path, image_size)
    YOLOv3.load_weights(weight_path)
    yolo_module_list = list(YOLOv3.children())[0]

    measurer = YOLOMeasurer(YOLOv3, yolo_module_list)
