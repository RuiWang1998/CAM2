import torch
import torch.nn as nn
import torch.optim.adam as Adam
import torch.distributions as distribution


class MultiStepVisualizer:
    """
    This should serve as a framework for YOLO and later some other models
    """
    def __init__(self, module_list, initial_size=30, model_intake_size=416, upscale_step=12,
                 batch_size=1, channel_num=3, generate_mean=0, generate_std=1, device='cpu'):
        """
        This function initializes the visualizer
        :param module_list: the module list from the model
        :param initial_size: the initial image size
        :param model_intake_size: the model's input image size
        :param upscale_step: the number of steps to take to upscale an image
        :param batch_size: the batch size of the image to optimize
        :param channel_num: the number of channels
        :param generate_mean: the mean of the normal distribution to generate new images
        :param generate_std: the standard deviation of the normal distribution to generate new images
        :param device: the device to put the model and data on
        """
        self.module_list = module_list
        self.step_idx = 0                       # the current step's index
        self.ini_size = initial_size
        self.input_size = model_intake_size
        self.upscale_step = upscale_step
        self.batch_size = batch_size
        self.z_image = self.image_init(batch_size, channel_num, model_intake_size)  # initialize the first image
        # the sampler to generate new images
        self.sampler = distribution.Normal(self.cast(generate_mean), self.cast(generate_std))
        self.current_size = self.ini_size   # the current size of the image

    @staticmethod
    def cast(value, d_type=torch.float32):
        """
        This function takes care of the casting because it is too tedious in Pytorch
        :param value: the number to convert
        :param d_type: the data type
        :return: the cast value
        """
        return torch.tensor(value).type(d_type)

    def image_init(self, batch_size, channel_num, model_intake_size):
        """
        This function initializes a new original image
        :param batch_size: the batch size of the image
        :param channel_num: the number of channels
        :param model_intake_size: the image size of the model
        :return: the image initialized
        """
        batch_size = self.cast(batch_size)
        channel_num = self.cast(channel_num)
        model_intake_size= self.cast(model_intake_size)
        if not isinstance(model_intake_size, list):
            return self.sampler.sample((batch_size, channel_num, model_intake_size, model_intake_size))
        else:
            return self.sampler.sample((batch_size, channel_num, model_intake_size[0], model_intake_size[1]))


