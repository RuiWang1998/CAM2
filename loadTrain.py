import torch

from models import Generator, Discriminator
from constants import D_PATH, G_PATH, device
from function import generator_train_step, discriminator_train_step
from data_load import day_loader, nig_loader
from main import d_optimizer, g_optimizer

generator = Generator().to(device)
generator.load_state_dict(torch.load(G_PATH))

discriminator = Discriminator().to(device)
discriminator.load_state_dict(torch.load(D_PATH))

minimax_loss = torch.nn.BCELoss()

if __name__ == '__main__':

    for epoch in range(2000):

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