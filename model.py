from collections import OrderedDict
import torch
import torch.nn as nn
from torchdiffeq import odeint

import h5py
import glob
import numpy as np
import time

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from numpy.random import default_rng

from collections import OrderedDict
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, nhidden, output_dim, nlayers=3, activation_fn = "Tanh", final_activation=None, scale_output=None):
        super(MLP, self).__init__()

        act_fn = getattr(nn, activation_fn)
        assert act_fn.__module__ == 'torch.nn.modules.activation', f'{activation_fn} is not a module of torch.nn.modules.activation'
        if final_activation is not None:
            final_act_fn = getattr(nn, final_activation)
            assert final_act_fn.__module__ == 'torch.nn.modules.activation', f'{final_activation_fn} is not a module of torch.nn.modules.activation'



        self.layers = [nn.Linear(input_dim, nhidden), act_fn()]
        for i in range(nlayers-2):
            self.layers.append(nn.Linear(nhidden, nhidden))
            self.layers.append(act_fn())
        self.layers.append(nn.Linear(nhidden, output_dim))

        if scale_output is not None:
            assert final_activation is None, 'final activation has to be none if we scale the output with scaling factor'

        if final_activation is not None:
            self.layers.append(final_act_fn())

        self.net = nn.Sequential(*self.layers)
        self.scale_output = scale_output

        if self.scale_output is not None:
            self.scaling_factor = nn.Parameter(torch.zeros(output_dim))
    def forward(self,x):
        out = self.net(x)
        if self.scale_output is None:
            return out
        s_fac = self.scaling_factor.exp().view(1, -1)
        s = torch.tanh(out / s_fac) * s_fac
        return s

class binaryODE(torch.nn.Module):
    def __init__(self, nhidden):
        super().__init__()
        self.nhidden = nhidden
        self.linear1 = torch.nn.Linear(nhidden*nhidden, nhidden)

    def forward(self, t, z):
        quad_z = torch.einsum("bp, bq->bpq" , z, z).reshape(-1, self.nhidden* self.nhidden)
        return self.linear1(quad_z)

class ODEFunc(nn.Module):
    def __init__(self, latent_dim, nhidden, nlayers=3, activation_fn = "Tanh", final_activation=None):
        super(ODEFunc, self).__init__()
        self.rhs_func = MLP(latent_dim, nhidden, latent_dim, nlayers=nlayers, activation_fn = activation_fn, final_activation=None, scale_output=True)

    def forward(self, t, x):
        return self.rhs_func(x)



class LatentODEfunc(nn.Module):
    def __init__(self,log_mean, log_std, latent_dim=3, nlayers=4, nhidden=20, mixing=1.0, nspecies=10, use_binaryODE=False, activation_fn = 'SiLU', ode_final_activation=None):
        super(LatentODEfunc, self).__init__()

        self.decoder = Decoder(latent_dim*2 + nspecies, nhidden, nspecies, log_mean, log_std, activation_fn=activation_fn, nlayers=nlayers)
        self.encoder = Encoder(nspecies, nhidden, latent_dim, log_mean, log_std, activation_fn=activation_fn, nlayers=nlayers)

        if use_binaryODE:
            self.rhs_func = binaryODE(latent_dim)
        else:
            self.rhs_func = ODEFunc(latent_dim, nhidden, nlayers=nlayers, activation_fn=activation_fn, final_activation =ode_final_activation)
        self.nfe = 0

        self.mixing =mixing
        self.nspecies = nspecies
        self.latent_dim = latent_dim

        self.log_mean = torch.from_numpy(log_mean)
        self.log_std  = torch.from_numpy(log_std)

    def encode_initial_condition(self, x0):
        return self.encoder(x0)

    def integrate_ODE(self, t, z0):
        return odeint(self.rhs_func, z0, t)

    def decode_path(self, z_path, z0, x0):
        ntimesteps = z_path.shape[0]
        X0_repeat = x0.unsqueeze(1).repeat(1,ntimesteps,1).permute(1,0,2)
        z0_repeat = z0.unsqueeze(1).repeat(1,ntimesteps,1).permute(1,0,2)

        # decode a normalize value!
        x_pred = self.decoder(z_path, z0_repeat, X0_repeat).permute(1,2,0)

        return x_pred

    def reconstruction_loss(self, x, x0, z0):
        ntimesteps = x.shape[2]
        X0_repeat = x0.unsqueeze(2).repeat(1,1,ntimesteps).permute(0,2,1)
        z0_repeat = z0.unsqueeze(2).repeat(1,1,ntimesteps).permute(0,2,1)

        z = self.encoder(x.permute(0,2,1))
        reconstruct_x = self.decoder(z, z0_repeat, X0_repeat)

#real_x = self.encoder.log_normalize(x.permute(0,2,1))
#loss_recon =  ((self.encoder.log_normalize(reconstruct_x[:,0,:]) - real_x[:,0,:])).abs().mean()

        # full autoencoder loss
        real_x = self.encoder.log_normalize(x.permute(0,2,1))
        loss_recon =  ((self.encoder.log_normalize(reconstruct_x) - real_x)).abs().mean()


        return loss_recon

    def forward(self, t, x):
        x0 = x[:,:,0]

        z0     = self.encode_initial_condition(x0)
        z_path = self.integrate_ODE(t[0], z0)
        x_pred = self.decode_path(z_path, z0, x0)

        real_x = self.encoder.log_normalize(x.reshape(-1,self.nspecies))
        pred_x = self.encoder.log_normalize(x_pred.reshape(-1,self.nspecies))

        loss_path = ((pred_x -real_x)).abs().mean()

        loss_recon = self.reconstruction_loss(x, x0, z0)

        #z_path_enc  = self.encoder(x.permute(2,0,1))
        #loss_recon += 0.01*((z_path_enc - z_path.detach())).abs().mean()


        #x_pred_detach = self.decode_path(z_path.detach(), z0, x0)
        #pred_x_detach = self.encoder.log_normalize(x_pred_detach.reshape(-1,self.nspecies))
        #loss_recon += ((pred_x_detach -real_x)).abs().mean()


        return x_pred, loss_path, loss_recon

    def plot_example(self,t, X, savefig=None):
        f,axes = plt.subplots(2,5,figsize=(20,10))
        with torch.no_grad():
            abund_paths, loss_path, loss_recon = self(t,X)
        for i in range(10):
            axes.flat[i].loglog(t[0], X[0,i,:].detach())
            axes.flat[i].plot(t[0], abund_paths[0,i,:].detach(), marker='.')
        plt.tight_layout()
        if savefig is not None:
            plt.savefig(savefig)
        plt.show()

class PlainDecoder(torch.nn.Module):
    """Decoder maps latent state to the actual observations
    We need to also account for the normalization done to the data.

    We constraint the decoder to output positive definite output

    """
    def __init__(self, latent_dim, nhidden, output_size, data_mean, data_std, nlayers=3, activation_fn='Tanh'):
        super(PlainDecoder, self).__init__()

        self.net = MLP(latent_dim, nhidden, output_size, nlayers=nlayers, activation_fn = activation_fn)

        self.mean = torch.from_numpy(data_mean)
        self.std  = torch.from_numpy(data_std)

    def log_normalize(self,x):
        return (torch.log10(x) - self.mean) / self.std

    def undo_log_normalize(self,x):
        return torch.pow(10, x* self.std + self.mean)

    def forward(self, z):
        return self.undo_log_normalize(self.net(z))

class Decoder(torch.nn.Module):
    """Decoder maps latent state to the actual observations
    We need to also account for the normalization done to the data.

    We constraint the decoder to output positive definite output

    """
    def __init__(self, latent_dim, nhidden, output_size, data_mean, data_std, nlayers=3, activation_fn='Tanh'):
        super(Decoder, self).__init__()

        self.net = MLP(latent_dim, nhidden, output_size, nlayers=nlayers, activation_fn = activation_fn)

        self.mean = torch.from_numpy(data_mean)
        self.std  = torch.from_numpy(data_std)
        self.scaling_factor = nn.Parameter(torch.zeros(output_size))

    def log_normalize(self,x):
        return (torch.log10(x) - self.mean) / self.std

    def undo_log_normalize(self,x):
        return torch.pow(10, x* self.std + self.mean)

    def forward(self, z, z0, x0):
        pos_output = self.net( torch.cat((z,z0,self.log_normalize(x0)),dim=-1 ))
        #return x0 * torch.exp(pos_output * self.std)
        s_fac = self.scaling_factor.exp().view(1, -1)
        return x0 * torch.exp(pos_output * s_fac) #* self.weights)

class Encoder(torch.nn.Module):
    def __init__(self, input_size, nhidden, latent_dim, data_mean, data_std, nlayers=3, activation_fn='Tanh'):
        super(Encoder, self).__init__()

        self.net = MLP(input_size, nhidden, latent_dim, nlayers=nlayers, activation_fn = activation_fn)
        self.mean = torch.from_numpy(data_mean)
        self.std  = torch.from_numpy(data_std)

    def log_normalize(self,x):
        return (torch.log10(x) - self.mean) / self.std

    def forward(self, x):
        norm_data = self.log_normalize(x)
        return self.net(norm_data)

class LatentODEfunc_PlainDec(nn.Module):
    def __init__(self,log_mean, log_std, latent_dim=3, nlayers=4, nhidden=20, mixing=1.0, nspecies=10, use_binaryODE=False, activation_fn = 'SiLU', ode_final_activation=None):
        super(LatentODEfunc_PlainDec, self).__init__()

        self.decoder = PlainDecoder(latent_dim, nhidden, nspecies, log_mean, log_std, activation_fn=activation_fn, nlayers=nlayers)
        self.encoder = Encoder(nspecies, nhidden, latent_dim, log_mean, log_std, activation_fn=activation_fn, nlayers=nlayers)

        if use_binaryODE:
            self.rhs_func = binaryODE(latent_dim)
        else:
            self.rhs_func = ODEFunc(latent_dim, nhidden, nlayers=nlayers, activation_fn=activation_fn, final_activation =ode_final_activation)
        self.nfe = 0

        self.mixing =mixing
        self.nspecies = nspecies
        self.latent_dim = latent_dim

        self.log_mean = torch.from_numpy(log_mean)
        self.log_std  = torch.from_numpy(log_std)

    def encode_initial_condition(self, x0):
        return self.encoder(x0)

    def integrate_ODE(self, t, z0):
        return odeint(self.rhs_func, z0, t)

    def decode_path(self, z_path, z0, x0):
        ntimesteps = z_path.shape[0]
        X0_repeat = x0.unsqueeze(1).repeat(1,ntimesteps,1).permute(1,0,2)
        z0_repeat = z0.unsqueeze(1).repeat(1,ntimesteps,1).permute(1,0,2)

        # decode a normalize value!
        x_pred = self.decoder(z_path, z0_repeat, X0_repeat).permute(1,2,0)

        return x_pred

    def reconstruction_loss(self, x, x0, z0):
        ntimesteps = x.shape[2]
        X0_repeat = x0.unsqueeze(2).repeat(1,1,ntimesteps).permute(0,2,1)
        z0_repeat = z0.unsqueeze(2).repeat(1,1,ntimesteps).permute(0,2,1)

        z = self.encoder(x.permute(0,2,1))
        reconstruct_x = self.decoder(z)

#real_x = self.encoder.log_normalize(x.permute(0,2,1))
#loss_recon =  ((self.encoder.log_normalize(reconstruct_x[:,0,:]) - real_x[:,0,:])).abs().mean()

        # full autoencoder loss
        real_x = self.encoder.log_normalize(x.permute(0,2,1))
        loss_recon =  ((self.encoder.log_normalize(reconstruct_x) - real_x)).abs().mean()


        return loss_recon

    def forward(self, t, x):
        x0 = x[:,:,0]

        z0     = self.encode_initial_condition(x0)
        z_path = self.integrate_ODE(t[0], z0)
        x_pred = self.decoder(z_path).permute(1,2,0)

        real_x = self.encoder.log_normalize(x.reshape(-1,self.nspecies))
        pred_x = self.encoder.log_normalize(x_pred.reshape(-1,self.nspecies))

        loss_path = ((pred_x -real_x)).abs().mean()

        loss_recon = self.reconstruction_loss(x, x0, z0)

        return x_pred, loss_path, loss_recon

    def plot_example(self,t, X, savefig=None):
        f,axes = plt.subplots(2,5,figsize=(20,10))
        with torch.no_grad():
            abund_paths, loss_path, loss_recon = self(t,X)
        for i in range(10):
            axes.flat[i].loglog(t[0], X[0,i,:].detach())
            axes.flat[i].plot(t[0], abund_paths[0,i,:].detach(), marker='.')
        plt.tight_layout()
        if savefig is not None:
            plt.savefig(savefig)
        plt.show()

