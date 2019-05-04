import torchvision.models as models

from visualizer import Visualizer

if __name__ == "__main__":
    vgg16 = models.vgg16(pretrained=True)
    vgg_visualizer = Visualizer(vgg16, list(vgg16.children()),
                                model_intake_size=224)

    vgg_visualizer.multistep_visualize(30, 180, data_path="vgg")
