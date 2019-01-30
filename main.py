import torch
import torch.nn as nn

from constants import EPOCHS, LATENT_DIM, device, LR
from data_load import day_loader, nig_loader
from models import Generator, Discriminator
from function import generator_train_step, discriminator_train_step

# The models
generator = Generator().to(device)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=LR)

discriminator = Discriminator().to(device)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR)

# Loss function
minimax_loss = nn.BCELoss()

for epoch in range(EPOCHS):

    d_loss = 0
    g_loss = 0
    for idx, ((images_day, _), (images_nig, _)) in enumerate(zip(day_loader, nig_loader)):
        batch_size_idx = images_day.shape[0]
        images_day, images_nig = images_day.to(device), images_nig.to(device)

        d_loss += discriminator_train_step(generator, images_day, images_nig, discriminator, batch_size_idx, minimax_loss, d_optimizer)

        g_loss += generator_train_step(generator, images_nig, discriminator, batch_size_idx, minimax_loss, g_optimizer)

    print("Generator loss:{:5f}| Discriminator:{:5f}".format(g_loss, d_loss))
