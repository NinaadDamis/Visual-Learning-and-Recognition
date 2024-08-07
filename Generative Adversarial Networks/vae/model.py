import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        """
        TODO 2.1.1 : Fill in self.convs following the given architecture 
         Sequential(
                (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): ReLU()
                (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                (3): ReLU()
                (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                (5): ReLU()
                (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            )
        """
        self.convs= nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                    )
        self.conv_out_dim = input_shape[1] // 8 * input_shape[2] // 8 * 256
        # print("Encoder Conv out dim = ", self.conv_out_dim)

        # TODO 2.1.1: fill in self.fc, such that output dimension is self.latent_dim
        self.fc = nn.Linear(self.conv_out_dim, self.latent_dim)

    def forward(self, x):
        # TODO 2.1.1 : forward pass through the network, output should be of dimension : self.latent_dim
        out = self.convs(x)
        # print("Encoder conv out shape = ", out.shape)
        out = out.view(-1,self.conv_out_dim)
        # print("Encoder conv out shape = ", out.shape)
        out = self.fc(out)
        # print("Forward output shape {0} == Latent dim shape {1}".format(out.shape,self.latent_dim))
        return out


class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        # TODO 2.2.1: fill in self.fc, such that output dimension is 2*self.latent_dim
        self.fc = nn.Linear(self.conv_out_dim, 2*self.latent_dim)
    
    def forward(self, x):
        # TODO 2.2.1: forward pass through the network.
        # should return a tuple of 2 tensors, each of dimension self.latent_dim

        out = self.convs(x)
        # print("Encoder conv out shape = ", out.shape)
        out = out.view(-1,self.conv_out_dim)
        # print("Encoder conv out shape = ", out.shape)
        out = self.fc(out)
        # print("Forward output shape {0} == Latent dim shape {1}".format(out.shape,self.latent_dim))
        # split_out = out.view(-1,self.latent_dim)
        # print("Split out shape = ", split_out.shape)
        mu = out[:,0:self.latent_dim]
        # print("Mu shape = ", mu.shape)
        sigma = out[:,self.latent_dim:2*self.latent_dim]
        # print("sigma shape = ", sigma.shape)
        tuple_out = (mu,sigma)
        return tuple_out

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        #TODO 2.1.1: fill in self.base_size
        self.base_size = np.array([4,4,128])
        self.fc = nn.Linear(latent_dim, np.prod(self.base_size))
        
        """
        TODO 2.1.1 : Fill in self.deconvs following the given architecture 
        Sequential(
                (0): ReLU()
                (1): ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (2): ReLU()
                (3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (4): ReLU()
                (5): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (6): ReLU()
                (7): Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        """
        self.deconvs = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )

    def forward(self, z):
        # TODO 2.1.1: forward pass through the network, first through self.fc, then self.deconvs.
        out = self.fc(z)
        # print("Decoder linear out = ", out.shape)
        out = out.view(-1,128,4,4)
        # print("Decoder linear reshape = ", out.shape)
        out = self.deconvs(out)
        # print("Decoder shape output = {0} should be equal to output shape {1}".format(out.shape,self.output_shape))
        return out

class AEModel(nn.Module):
    def __init__(self, variational, latent_size, input_shape = (3, 32, 32)):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        if variational:
            self.encoder = VAEEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)