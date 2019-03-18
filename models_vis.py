import torch.nn as nn


class Rectifier(nn.Module):
    def __init__(self):
        super(Rectifier, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 20, 4, 1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 1, 6, 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.MaxPool2d(7)
        )

        self.linear1 = nn.Linear(324, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):

        out = self.conv1(img)
        out = self.conv2(out)

        out = out.view(img.shape[0], -1)

        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)

        return out
