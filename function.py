import matplotlib.pyplot as plt
import torch

from cgan_constants import LATENT_DIM, device


def generator_train_step(generator, images_nig, discriminator, batch_size, loss_func, optimizer):
    """
    trains the generator
    :param generator: the generator network to train
    :param images_nig: the night image
    :param discriminator: the discriminator
    :param batch_size: the size of the mini batch
    :param loss_func: the loss function to train with, typically BCE
    :param optimizer: the optimizer
    :return: the loss value
    """
    optimizer.zero_grad()

    z = torch.randn(batch_size, LATENT_DIM).to(device)
    fake_day_images = generator(z, images_nig)
    score = discriminator(images_nig, fake_day_images)
    loss = loss_func(score, torch.ones(batch_size).to(device))

    loss.backward()
    optimizer.step()

    return loss


def discriminator_train_step(generator, images_day, images_nig, discriminator, batch_size, loss_func, optimizer):
    """
    trains the generator
    :param generator: the generator network to train
    :param images_day: the day time image
    :param images_nig: the night time image
    :param discriminator: the discriminator
    :param batch_size: the size of the mini batch
    :param loss_func: the loss function to train with, typically BCE
    :param optimizer: the optimizer
    :return: the loss value
    """
    optimizer.zero_grad()
    # real images pairs
    score = discriminator(images_nig, images_day)
    loss_real = loss_func(score, torch.ones(batch_size).to(device))

    # fake image pairs
    z = torch.randn(batch_size, LATENT_DIM).to(device)
    fake_day_images = generator(z, images_nig)
    score = discriminator(images_nig, fake_day_images)
    loss_fake = loss_func(score, torch.zeros(batch_size).to(device))

    # backward pass
    loss = loss_real + loss_fake
    loss.backward()
    optimizer.step()

    return loss


def visualize(generator, images_nig):
    generator.eval()
    z = torch.randn(1, LATENT_DIM).to(device)
    image_fake = generator(z, torch.unsqueeze(images_nig[0], 0).to(device))
    image_fake = torch.squeeze(image_fake).permute(1, 2, 0).detach().cpu()
    plt.imshow(image_fake)
    plt.show()


def save(model, path):
    torch.save(model.state_dict(), path)
