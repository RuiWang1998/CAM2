import torch


class SensitivityMeasurer:
    """
    This class serves as a sensitivity measurer that has two metrics from the paper
    Sensitivity and Generalization in Neural Networks: an Empirical Study
    """
    def __init__(self, model, module_list=None, cuda=True, model_intake_size=416, batch_size=1, channel_num=3):
        """
        This function initializes the object
        :param model: the model of interest
        :param module_list: the module list of the model, if provided, otherwise will be written by list(model.children())
        :param cuda: whether to use cuda, which will be limited by hardware
        :param model_intake_size: the input size of the image, could be a int or a list/tuple
        :param batch_size: the batch size of the input
        :param channel_num: the number of input channel
        """
        # check if use cuda
        self.cuda = cuda and torch.cuda.is_avaiable()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # get the models and module list
        if model is not None:
            self.model = model.to(self.device)
        if module_list is not None:
            self.module_list = module_list
        else:
            self.module_list = list(model.children())

        self.layer_num = len(self.module_list)  # the number of layers of the model
        self.channel_count = []  # the number of channels of each layer
        self.batch_size = batch_size  # the batch size of the input to the model
        if isinstance(model_intake_size, int):
            self.width = model_intake_size
            self.height = model_intake_size
        else:
            self.width, self.height = model_intake_size
        self.channel_num = channel_num

    def get_n_th_layer_act_core(self, img, layer_idx):
        """
        This function gets the nth layer output
        :param img: the input image
        :param layer_idx: the layer index of interest
        :return: the output at that layer
        """
        for i, layer in enumerate(self.module_list):
            try:
                img = layer(img)
            except NotImplementedError:
                pass
            if i == layer_idx:
                return img
        raise ValueError(f"Layer {layer_idx} is larger than the maximum layer count {self.layer_num}")

    @staticmethod
    def reduction(reduction_tensor, reduction_type):
        """
        This function does the dimensionality reduction
        :param reduction_tensor: the tensor to reduce dimensionality of
        :param reduction_type: the type of reduction to perform, one of 'mean' or 'sum'
        :return: the reduced tensor
        """
        if reduction_type is None:
            return reduction_tensor
        if reduction_type == 'mean':
            return reduction_tensor.mean()
        if reduction_type == 'sum':
            return reduction_tensor.sum()

    def get_n_th_layer_act(self, img, layer_idx, reduction=None):
        """
        This function wraps around the get_n_th_layer_act_core function to introduce the ability of reduction
        :param img: the input image
        :param layer_idx: the the index of the layer of interest
        :param reduction: the reduction method, one of 'sum' or 'mean'
        :return: the n-th layer's activation
        """
        return self.reduction(self.get_n_th_layer_act_core(img, layer_idx), reduction)

    def get_nth_channel(self, img, layer_idx, channel_idx, reduction=None):
        """
        This function gets the n-th channel of the n-th layer's output
        :param img: the input image
        :param layer_idx: the layer index
        :param channel_idx: the channel index
        :param reduction: the reduction method, one of 'sum' or 'mean'
        :return: the n-th channel output
        """
        return self.reduction(self.get_n_th_layer_act(img, layer_idx)[:, channel_idx, :, :], reduction)

    def get_nth_neuron(self, img, layer_idx, channel_idx, neuron_idx):
        """
        This function gets the n-th neuron of the n-th channel of the n-th layer's output
        :param img: the input image
        :param layer_idx: the index of the layer of interest
        :param channel_idx: the index of the channel of interest
        :param neuron_idx: the index or the coordinate of the neuron of interest
        :return: the activation of that neuron
        """
        if isinstance(neuron_idx, list) or isinstance(neuron_idx, tuple):
            x_idx, y_idx = neuron_idx
        else:
            x_idx = y_idx = neuron_idx
        return self.get_nth_channel(img, layer_idx, channel_idx, reduction=None)[x_idx, y_idx]
