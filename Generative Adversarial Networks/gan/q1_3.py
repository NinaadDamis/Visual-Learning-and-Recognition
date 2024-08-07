import os

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    # TODO 1.3.1: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    
    loss_fn     = torch.nn.BCEWithLogitsLoss()
    # Label = 1 for real data
    real_labels = torch.ones_like(discrim_real)
    # Label = 0 for generated data
    fake_labels = torch.zeros_like(discrim_fake)

    stacked_out    = torch.hstack((discrim_real,discrim_fake))
    stacked_labels = torch.hstack((real_labels,fake_labels))
    
    loss = loss_fn(stacked_out,stacked_labels)

    return loss


def compute_generator_loss(discrim_fake):
    # TODO 1.3.1: Implement GAN loss for generator.
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # Labels 1 for true data of correct shape
    labels  = torch.ones_like(discrim_fake)
    loss    = loss_fn(discrim_fake,labels)

    return loss


if __name__ == "__main__":
    gen = Generator().cuda().to(memory_format=torch.channels_last)
    disc = Discriminator().cuda().to(memory_format=torch.channels_last)
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.3.2: Run this line of code.
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
