from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from model import AEModel
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import time
import os
from utils import *
import wandb


def ae_loss(model, x):
    """ 
    TODO 2.1.2: fill in MSE loss between x and its reconstruction. 
    return loss, {recon_loss = loss} 
    """
    batch_size = x.shape[0]
    encoder_out = model.encoder(x)
    decoder_out = model.decoder(encoder_out)
    loss = nn.MSELoss(reduction='sum')
    mean_loss  = loss(x,decoder_out) / batch_size
    # print("Mean loss = ", mean_loss)
    # mean_loss = np.mean(loss_arr)
    return mean_loss, OrderedDict(recon_loss=mean_loss)

def vae_loss(model, x, beta = 1):
    """TODO 2.2.2 : Fill in recon_loss and kl_loss. """

    recon_loss_fn = nn.MSELoss(reduction='sum')
    # kl_loss_fn    = nn.KLDivLoss(reduction='batchmean')
    batch_size    = x.shape[0]
    encoder_out   = model.encoder(x)

    latent = encoder_out[0] + torch.exp(encoder_out[1]) * torch.randn(size=encoder_out[1].shape).cuda()

    variance = torch.exp(encoder_out[1]) ** 2
    variance_log = torch.log(variance)
    kl_loss = 0.5 * torch.mean(torch.sum(encoder_out[0] ** 2 + variance - variance_log - 1, dim=1))

    # latent = torch.from_numpy(latent).float().cuda()
    decoder_out   = model.decoder(latent)
    # print("Decoder out shape = ", decoder_out.shape)
    recon_loss    = recon_loss_fn(x,decoder_out) / batch_size
    # print("recon loss shape = ", recon_loss.shape)

    total_loss = recon_loss + beta*kl_loss
    return total_loss, OrderedDict(recon_loss=recon_loss, kl_loss=kl_loss)


def constant_beta_scheduler(target_val = 1):
    def _helper(epoch):
        return target_val
    return _helper

def linear_beta_scheduler(max_epochs=None, target_val = 1):
    """TODO 2.3.2 : Fill in helper. The value returned should increase linearly 
    from 0 at epoch 0 to target_val at epoch max_epochs """
    step = target_val / (max_epochs-1)
    betas = [step*i for i in range(max_epochs)]
    def _helper(epoch):
       return betas[epoch]
    return _helper

def run_train_epoch(model, loss_mode, train_loader, optimizer, beta = 1, grad_clip = 1):
    model.train()
    all_metrics = []
    for x, _ in train_loader:
        x = preprocess_data(x)
        if loss_mode == 'ae':
            loss, _metric = ae_loss(model, x)
        elif loss_mode == 'vae':
            loss, _metric= vae_loss(model, x, beta)
        all_metrics.append(_metric)
        optimizer.zero_grad()
        # print("Loss before backward = ", loss.shape,loss)
        loss.backward()
        
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return avg_dict(all_metrics)

def get_val_metrics(model, loss_mode, val_loader):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = preprocess_data(x)
            if loss_mode == 'ae':
                loss, _metric = ae_loss(model, x)
            elif loss_mode == 'vae':
                loss, _metric= vae_loss(model, x)
            all_metrics.append(_metric)

    return avg_dict(all_metrics)

def main(log_dir, loss_mode = 'vae', beta_mode = 'constant', num_epochs = 20, batch_size = 256, latent_size = 256,
         target_beta_val = 1, grad_clip=1, lr = 1e-3, eval_interval = 5):

    os.makedirs('data/'+ log_dir, exist_ok = True)
    train_loader, val_loader = get_dataloaders()

    variational = True if loss_mode == 'vae' else False
    model = AEModel(variational, latent_size, input_shape = (3, 32, 32)).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    vis_x = next(iter(val_loader))[0][:36]
    loss_arr = []
    recon_loss_arr = []
    kl_loss_arr = []
    
    #beta_mode is for part 2.3, you can ignore it for parts 2.1, 2.2
    if beta_mode == 'constant':
        beta_fn = constant_beta_scheduler(target_val = target_beta_val) 
    elif beta_mode == 'linear':
        beta_fn = linear_beta_scheduler(max_epochs=num_epochs, target_val = target_beta_val) 

    for epoch in range(num_epochs):
        print('epoch', epoch)
        train_metrics = run_train_epoch(model, loss_mode, train_loader, optimizer, beta_fn(epoch))
        val_metrics = get_val_metrics(model, loss_mode, val_loader)
        # print("VAL METRICS = ", val_metrics)
        if loss_mode == 'ae':
            loss_arr.append(val_metrics['recon_loss'])
        if loss_mode == 'vae':
            recon_loss_arr.append(val_metrics['recon_loss'])
            kl_loss_arr.append(val_metrics['kl_loss'])
            wandb.log({'validation/recon_loss': val_metrics['recon_loss']})
            wandb.log({'validation/kl_loss': val_metrics['kl_loss']})
        #TODO : add plotting code for metrics (required for multiple parts)

        if (epoch+1)%eval_interval == 0:
            print(epoch, train_metrics)
            print(epoch, val_metrics)

            vis_recons(model, vis_x, 'data/'+log_dir+ '/epoch_'+str(epoch))
            if loss_mode == 'vae':
                vis_samples(model, 'data/'+log_dir+ '/epoch_'+str(epoch) )

    # if loss_mode == 'ae':
    return loss_arr
    # else:
    #     return recon_loss_arr, kl_loss_arr

if __name__ == '__main__':
    pass
    #TODO: Experiments to run : 
    #2.1 - Auto-Encoder
    #Run for latent_sizes 16, 128 and 1024
    wandb.init(project="VLR2 - VAE ", reinit=True)
    loss_arr_1024 = main('ae_latent1024', loss_mode = 'ae',  num_epochs = 20, latent_size = 1024)
    loss_arr_128 = main('ae_latent128', loss_mode = 'ae',  num_epochs = 20, latent_size = 128)
    loss_arr_16 = main('ae_latent16', loss_mode = 'ae',  num_epochs = 20, latent_size = 16)

    columns = ['Latent size = 1024', 'Latent size = 128', 'Latent size = 16']
    num_steps = 20
    xs = [ i for i in range(num_steps) ]
    ys= [loss_arr_1024,loss_arr_128,loss_arr_16]
    wandb.log({"Autoencoder Reconstruction Loss" : wandb.plot.line_series(
    xs=xs,
    ys=ys,
    keys=columns,
    title="Reconstruction Loss VS Epochs")})

    # Q 2.2 - Variational Auto-Encoder
    loss_arr= main('vae_latent1024', loss_mode = 'vae', num_epochs = 20, latent_size = 1024)

    # Q 2.3.1 - Beta-VAE (constant beta)
    # Run for beta values 0.8, 1.2
    main('vae_latent1024_beta_constant1.2', loss_mode = 'vae', beta_mode = 'constant', target_beta_val = 1.2, num_epochs = 20, latent_size = 1024)

    #Q 2.3.2 - VAE with annealed beta (linear schedule)
    main(
        'vae_latent1024_beta_linear1', loss_mode = 'vae', beta_mode = 'linear', 
        target_beta_val = 1, num_epochs = 20, latent_size = 1024
    )