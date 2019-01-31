#%%
import torch
import torch.nn as nn
import time

from constants import EPOCHS, LATENT_DIM, device, LR_G, LR_D, G_PATH, D_PATH
from data_load import day_loader, nig_loader
from models import Generator, Discriminator
from function import generator_train_step, discriminator_train_step, visualize

if __name__ == '__main__':
            
    # The models   
    generator = Generator().to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=LR_G)

    discriminator = Discriminator().to(device)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR_D)
    
    # Loss function
    minimax_loss = nn.BCELoss() 
    
    for epoch in range(EPOCHS):

        d_loss = 0
        g_loss = 0
        generator.train()
        start_time = time.time()
        for idx, ((images_day, _), (images_nig, _)) in enumerate(zip(day_loader, nig_loader)):
            batch_size_idx = images_day.shape[0]
            images_day, images_nig = images_day.to(device), images_nig.to(device)

            d_loss += discriminator_train_step(generator, images_day, images_nig, discriminator, batch_size_idx, minimax_loss, d_optimizer)

            g_loss += generator_train_step(generator, images_nig, discriminator, batch_size_idx, minimax_loss, g_optimizer)
        if epoch % 3 == 0: pass # visualize(generator, images_nig)
        print("Epoch: {}| Generator loss:{:5f}| Discriminator:{:5f}| time elapsed:{:2f}".format(epoch, g_loss, d_loss, time.time() - start_time))
        if epoch % 200 == 0:
            torch.save(generator.state_dict(), G_PATH)
            torch.save(discriminator.state_dict(), D_PATH)

#%%
