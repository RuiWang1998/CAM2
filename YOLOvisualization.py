import torch

from YOLOv3.models import Darknet
from visualizer import MultiStepVisualizer


class YOLOv3Visualizer(MultiStepVisualizer):
    """
    This is a class that allows for some modifications for YOLOv3
    """

    def __init__(self, model, module_list, cuda=True):
        """
        First let us inherit the visualizer
        :param model: the YOLOv3 model pre-trained
        :param cuda: the device to work on
        """
        super(YOLOv3Visualizer, self).__init__(model, module_list=module_list, cuda=cuda)

    def get_nth_output_layer(self, img, layer_idx):
        """
        This function gets the n-th layer output of YOLOv3
        :param img: the input image
        :param layer_idx: the index of the layer
        :return: the output of the specific layer
        """
        return self.model(img, layer_idx=[layer_idx])[layer_idx]

    def _module_list_to_device(self):
        """
        This overloads the module list method, which modifies the self.channel_count
        """
        place_holder = self.random_init(self.input_size).to(device)
        outputs = self.model(place_holder, layer_idx=list(range(self.layer_num)))
        self.layer_num = len(set(outputs.values()))
        self.channel_count = [output.shape[1] for output in outputs.values()]
        del place_holder
        if self.cuda:
            self.clear_cuda_memory()


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image_size = 416
    config_path = 'YOLOv3/config/yolov3.cfg'
    weight_path = "YOLOv3/weights/yolov3.weights"
    image_folder = "YOLOv3/data/samples"
    class_path = 'YOLOv3/data/coco.names'
    batch_size = 1
    n_cpu = 8

    YOLOv3 = Darknet(config_path, image_size)
    YOLOv3.load_weights(weight_path)
    yolo_module_list = list(YOLOv3.children())[0]

    visualizer = YOLOv3Visualizer(YOLOv3, module_list=yolo_module_list, cuda=True)
    visualizer.visualize(10, 0, data_path='visualization', weight_decay=1e-5)
    ####
    visualizer.visualize_whole_layer(10, data_path='visualization', weight_decay=1e-5)
