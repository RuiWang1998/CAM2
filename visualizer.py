import math
import os

import torch
import torch.distributions as distribution
import torch.nn as nn
import torch.optim as optimizers
from scipy.misc import imsave as save_img


class MultiStepVisualizer:
    """
    This should serve as a framework for any sort of models and later some other models
    """

    def __init__(self, model, module_list=None, initial_size=30, model_intake_size=416, upscale_step=12,
                 batch_size=1, channel_num=3, device=torch.device('cpu')):
        """
        This function initializes the visualizer
        :param module_list: the module list from the model
        :param initial_size: the initial image size
        :param model_intake_size: the model's input image size
        :param upscale_step: the number of steps to take to upscale an image
        :param batch_size: the batch size of the image to optimize
        :param channel_num: the number of channels
        :param device: the device to put the model and data on
        """
        if model is not None:
            self.model = model.to(self.device)
        if module_list is not None:
            self.module_list = module_list
        else:
            self.module_list = list(model.children())

        self.layer_num = len(self.module_list)  # the number of layers in a model
        self.channel_count = []
        self._module_list_to_device()

        self.input_size = model_intake_size
        self.upscale_step = upscale_step
        self.batch_size = batch_size
        self.channel_num = channel_num  # number of channels in the input image
        self.z_image = self.image_init(initial_size)  # initialize the first image
        self.z_image.requires_grad = True
        # the sampler to generate new images
        self.current_size = initial_size  # the current size of the image
        # get the scale
        self.upscale_ratio = (self.input_size / initial_size) ** (1 / self.upscale_step)
        if isinstance(model_intake_size, int):
            self.input_generator = nn.UpsamplingNearest2d(size=(model_intake_size, model_intake_size))
        else:
            self.input_generator = nn.UpsamplingNearest2d(size=model_intake_size)

        self.up_scaler = nn.UpsamplingNearest2d(self.upscale_ratio)

        self.device = device

    def _module_list_to_device(self):
        """
        This function counts the number of channels in each layer
        """
        place_holder = self.random_init(self.input_size)
        for i, layer in enumerate(self.module_list):
            self.module_list[i] = layer.to(self.device)
            try:
                place_holder = self.module_list[i](place_holder)
                output_shape = place_holder.shape
                self.channel_count.append(output_shape[1])
            except NotImplementedError:
                self.channel_count.append(0)

    @staticmethod
    def cast(value, d_type=torch.float32):
        """
        This function takes care of the casting because it is too tedious in Pytorch
        :param value: the number to convert
        :param d_type: the data type
        :return: the cast value
        """
        return torch.tensor(value).type(d_type)

    @staticmethod
    def tanh(x, scaling=0.2):
        """
        This function put boundaries upon the image so that all of its values are of [0-1]
        :param x: the input
        :param scaling: the scaling of the tanh function
        :return: the scaled input
        """
        return torch.nn.Tanh()(scaling * x) / 2 + 0.5

    def random_init(self, image_size, mean=0, std=1):
        """
        This function initializes a new original image
        :param image_size: the image size of the model
        :param mean: the mean of the distribution
        :param std: the standard deviation of the distribution
        :return: the image initialized
        """
        image_size = self.cast(image_size)
        sampler = distribution.Normal(self.cast(mean), self.cast(std))

        if not isinstance(image_size, list):
            return sampler.sample((self.batch_size, self.channel_num, image_size, image_size)).to(self.device)
        else:
            return sampler.sample((self.batch_size, self.channel_num, image_size[0], image_size[1])).to(self.device)

    def noise_gen(self, image_size, noise_ratio=0.1, mean=0, std=1):
        """
        This function allows to create a mask onto
        :param self: the class itself
        :param image_size: the image size of the model
        :param noise_ratio: the noise ratio imposed onto the mask
        :param mean: the mean of the mask
        :param std: the standard deviation of the distribution
        :return: the random mask
        """
        return self.random_init(image_size, mean, std) * noise_ratio

    def image_init(self, image_size):
        """
        This function allows to create a mask onto
        :param self: the class itself
        :param image_size: the image size of the model
        :return: the random mask
        """
        image = self.random_init(image_size)
        image.require_grad = True

        return image

    def generate_input_image(self):
        """
        This function creates an input image for the model
        :return: the input image
        """
        return self.tanh(self.input_generator(self.z_image))

    def upscale_image(self, mask=True):
        """
        This function up-scales the current image
        :param mask: if to mask the up-scaled image
        """
        self.z_image = self.up_scaler(self.z_image)
        if mask:
            self.z_image = self.z_image + self.noise_gen()
        self.z_image.requires_grad = True

    @staticmethod
    def mkdir_single(path):
        """
        This function tries to create a file folder
        :param path: the path to create
        """
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

    def mkdir(self, data_path, layer_idx):
        """
        This function creates the folders to store the visualizations
        :param data_path: the data path to store in
        :param layer_idx: the layer number
        """
        self.mkdir_single(f"{data_path}")
        self.mkdir_single(f"{data_path}/layer{layer_idx}")
        self.mkdir_single(f"{data_path}/layer{layer_idx}")
        self.mkdir_single(f"{data_path}/layer{layer_idx}/Color")
        self.mkdir_single(f"{data_path}/layer{layer_idx}/Mono0")
        self.mkdir_single(f"{data_path}/layer{layer_idx}/Mono1")
        self.mkdir_single(f"{data_path}/layer{layer_idx}/Mono2")

    def save_image(self, data_path, layer_idx, channel_idx, epoch_idx):
        """
        This function saves the image
        :param data_path: the data path to save to
        :param layer_idx: the layer index of the images
        :param channel_idx: the channel index of the images
        :param epoch_idx: the epoch index this image is from
        """
        self.mkdir(data_path, layer_idx=layer_idx)

        img_to_save = self.generate_input_image()
        save_img(f"{data_path}/layer{layer_idx}/Color/channel{channel_idx}_epoch{epoch_idx}.jpg",
                 img_to_save.detach().cpu().permute(1, 2, 0)[:, :, :])
        save_img(f"{data_path}/layer{layer_idx}/Mono0/channel{channel_idx}_epoch{epoch_idx}.jpg",
                 img_to_save.detach().cpu().permute(1, 2, 0)[:, :, 0])
        save_img(f"{data_path}/layer{layer_idx}/Mono1/channel{channel_idx}_epoch{epoch_idx}.jpg",
                 img_to_save.detach().cpu().permute(1, 2, 0)[:, :, 1])
        save_img(f"{data_path}/layer{layer_idx}/Mono2/channel{channel_idx}_epoch{epoch_idx}.jpg",
                 img_to_save.detach().cpu().permute(1, 2, 0)[:, :, 2])

    def get_nth_output(self, img, layer_idx):
        """
        This function gets the n-th layer output
        :param img: the input image
        :param layer_idx: the layer index
        :return: the output
        """

        for i, layer in enumerate(self.module_list):
            try:
                img = layer(img)
            except NotImplementedError:
                pass
            if i == layer_idx:
                return img
        return img

    def visualize(self, layer_idx, channel_idx, epochs=30, optimizer=optimizers.Adam,
                  data_path=".", learning_rate=None, weight_decay=None):
        """
        This function does the visualization
        :param layer_idx: the index of the layer to visualize
        :param channel_idx: the channel index of the layer to visualize
        :param epochs: the number of epochs to train
        :param optimizer: the optimizer of the image
        :param data_path: the data path to store the images
        :param learning_rate: the learning rate of the optimizer
        :param weight_decay: the weight decay of the optimizer
        """
        if learning_rate is None:
            learning_rate = 0.001
        if weight_decay is None:
            weight_decay = 0

        for step in range(self.upscale_step):

            optimizer_instance = optimizer([self.z_image], lr=learning_rate, weight_decay=weight_decay)
            for epoch in range(epochs):
                optimizer_instance.zero_grad()

                img = self.generate_input_image()
                output = self.get_nth_output(img, layer_idx)
                output_channels = output.mean(-1).mean(-1).mean(0)
                activation = output_channels[channel_idx]

                activation.backward()
                optimizer_instance.step()
                # save image
                if epoch == math.floor(epochs / 2):
                    self.save_image(data_path, layer_idx, channel_idx, epoch)

            self.upscale_image()
            self.save_image(data_path, layer_idx, channel_idx, epochs)

    def visualize_all_model(self):
        """
        This function allows to visualize all the channels in a model
        """
        for layer_idx in range(self.layer_num):
            for channel_idx in self.channel_count[layer_idx]:
                self.visualize(layer_idx, channel_idx)
