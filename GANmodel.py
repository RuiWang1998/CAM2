import torch
import torch.nn as nn

from cgan_constants import BOTTLENEK, IMG_INPUT_LEN, LATENT_DIM, NUM_CHANNEL, PROCESSOR_OUT


class ImageProcessor(nn.Module):
    def __init__(self):
        super(ImageProcessor, self).__init__()

        def conv_layer(input_channel, output_channel):
            return nn.Sequential(
                nn.Conv2d(input_channel, output_channel, 3, 2, 1),
                nn.BatchNorm2d(output_channel),
                nn.ReLU()
            )

        self.conv1 = conv_layer(NUM_CHANNEL, 16)
        self.conv2 = conv_layer(16, 32)
        self.conv3 = conv_layer(32, 16)
        self.conv4 = conv_layer(16, 3)

        self.model = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4
        )

        self.out = nn.Sequential(
            nn.Linear(300, BOTTLENEK // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(BOTTLENEK // 2, BOTTLENEK // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(BOTTLENEK // 2, PROCESSOR_OUT))

    def forward(self, x):
        out = self.model(x)
        out = out.view(out.shape[0], -1)
        out = self.out(out)

        return out


class ImagePreProcessor(nn.Module):
    def __init__(self):
        super(ImagePreProcessor, self).__init__()

        def conv_layer(input_channel, output_channel):
            return nn.Sequential(
                nn.Conv2d(input_channel, output_channel, 3, 2, 1),
                nn.BatchNorm2d(output_channel),
                nn.ReLU()
            )

        self.conv1 = conv_layer(NUM_CHANNEL, 16)
        self.conv2 = conv_layer(16, 32)

        self.model = nn.Sequential(
            self.conv1,
            self.conv2,
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = IMG_INPUT_LEN // 4
        self.l1 = nn.Sequential(
            nn.Linear(LATENT_DIM + PROCESSOR_OUT, BOTTLENEK * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(BOTTLENEK),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(BOTTLENEK, BOTTLENEK, 3, stride=1, padding=1),
            nn.BatchNorm2d(BOTTLENEK, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(BOTTLENEK, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, NUM_CHANNEL, 3, stride=1, padding=2),
            nn.Tanh()
        )

        self.imageProcessor = ImageProcessor()

    def forward(self, z, x):
        proc = self.imageProcessor(x)
        out = self.l1(torch.cat((proc, z), dim=1))
        out = out.view(out.shape[0], BOTTLENEK, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img / 2 + 0.5


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.adv_layer = nn.Sequential(
            nn.Linear(300, BOTTLENEK // 2),
            nn.ReLU(),
            nn.Linear(BOTTLENEK // 2, BOTTLENEK // 2),
            nn.ReLU(),
            nn.Linear(BOTTLENEK // 2, 1),
            nn.Sigmoid())

        self.imageProcessor = ImagePreProcessor()

        def conv_layer(input_channel, output_channel):
            return nn.Sequential(
                nn.Conv2d(input_channel, output_channel, 3, 2, 1),
                nn.BatchNorm2d(output_channel),
                nn.ReLU()
            )

        self.conv1 = conv_layer(64, 128)
        self.conv2 = conv_layer(128, 3)

    def forward(self, img1, img2):
        out1 = self.imageProcessor(img1)
        out2 = self.imageProcessor(img2)
        out = torch.cat((out1, out2), dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        validity = self.adv_layer(out.view(img1.shape[0], -1))

        return validity
