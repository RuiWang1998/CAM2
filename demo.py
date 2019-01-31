import torch
import time
import matplotlib.pyplot as plt

from models import Generator
from data_load import day_loader, nig_loader
from constants import LATENT_DIM


if __name__ == '__main__':

    generator = Generator()
    generator.load_state_dict(torch.load('C:/New folder/Github/CAM2/CAM2/model/generator'))

    minimax_loss = torch.nn.BCELoss()
    for images, _ in nig_loader:pass
    image = images[0].permute(1,2,0)
    plt.imshow(image)
    plt.show()
    image = images[0:1]
    gen = generator(torch.randn(1, LATENT_DIM), image)
    plt.imshow(torch.squeeze(gen).permute(1,2,0).detach().numpy())
    plt.show()