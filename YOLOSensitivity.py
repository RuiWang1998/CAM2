import cv2
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

    def _module_list_channel_count(self):
        """
        This overloads the module list method, which modifies the self.channel_count
        """
        place_holder = torch.randn(self.batch_size, self.channel_num,
                                   self.height, self.width).to(self.device)
        outputs = self.model(place_holder, layer_idx=list(range(self.layer_num)))
        self.layer_num = len(set(outputs.values()))
        self.channel_count = [output.shape[1] for output in outputs.values()]
        self.size_count = [output.shape[2:] for output in outputs.values()]
        self.size_count[-1] = torch.Size([0, 0])
        del place_holder
        if self.cuda:
            torch.cuda.empty_cache()


if __name__ == '__main__':
    config_path = 'YOLOv3/config/yolov3.cfg'
    weight_path = "YOLOv3/weights/yolov3.weights"
    image_folder = "YOLOv3/data/samples"
    class_path = 'YOLOv3/data/coco.names'

    image_size = 416
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    img1 = cv2.imread('pair_compare/00073_000000463_.jpg')
    img1 = cv2.resize(img1, tuple([image_size, image_size]))
    img1 = torch.tensor(img1 / 255, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img2 = cv2.imread('pair_compare/00073_000000465_.jpg')
    img2 = cv2.resize(img2, tuple([image_size, image_size]))
    img2 = torch.tensor(img2 / 255, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    YOLOv3 = Darknet(config_path, image_size)
    YOLOv3.load_weights(weight_path)
    yolo_module_list = list(YOLOv3.children())[0]
    place_holder = torch.randn(1, 3, 416, 416).to(device)
    measurer = YOLOMeasurer(YOLOv3, yolo_module_list)
    # measurer.get_nth_neuron(place_holder, 1, 0, 0).shape
    # Jacob = measurer.compute_channel_jacobian(place_holder, 1, 0)
    # jacobian = measurer.compute_channel_jacobian(img1, [0, 0], "reduction_mean")
    mean_Jacobian1 = measurer.compute_jacobian(img1, mode="reduction_mean")
    mean_Jacobian2 = measurer.compute_jacobian(img2, mode="reduction_mean")
