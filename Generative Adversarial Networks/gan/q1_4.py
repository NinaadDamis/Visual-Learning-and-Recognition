import os

import torch
import torch.nn.functional as F

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    # TODO 1.4.1: Implement LSGAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    loss_fn = torch.nn.MSELoss()
    real_label = torch.ones(discrim_real.size(), dtype=discrim_real.dtype, layout=discrim_real.layout, device=discrim_real.device)
    fake_label = torch.zeros(discrim_fake.size(), dtype=discrim_fake.dtype, layout=discrim_fake.layout, device=discrim_fake.device)
    loss = (1/2) * (loss_fn(discrim_real,real_label) + loss_fn(discrim_fake,fake_label))

    return loss


def compute_generator_loss(discrim_fake):
    # TODO 1.4.1: Implement LSGAN loss for generator.
    loss_fn = torch.nn.MSELoss()
    labels = torch.ones(discrim_fake.size(), dtype=discrim_fake.dtype, layout=discrim_fake.layout, device=discrim_fake.device)
    loss = (1/2) * loss_fn(discrim_fake,labels)
    return loss


if __name__ == "__main__":
    gen = Generator().cuda().to(memory_format=torch.channels_last)
    disc = Discriminator().cuda().to(memory_format=torch.channels_last)
    prefix = "data_ls_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.4.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
    )
