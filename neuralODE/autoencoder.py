import torch
import time
from neuralODE.model import MLP
import numpy as np

import torch.nn as nn


class Encoder(torch.nn.Module):
    def __init__(self, input_size, nhidden, latent_dim, data_mean, data_std, nlayers=3, activation_fn='Tanh'):
        super(Encoder, self).__init__()

        self.net = MLP(input_size, nhidden, latent_dim, nlayers=nlayers, activation_fn = activation_fn)
        self.register_buffer('mean', torch.from_numpy(data_mean))
        self.register_buffer('std',  torch.from_numpy(data_std))
        #self.mean = torch.from_numpy(data_mean)
        #self.std  = torch.from_numpy(data_std),
        clip_value = 0.1
        for p in self.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    def log_normalize(self,x):
        return (torch.log10(x) - self.mean) / self.std

    def forward(self, x):
        norm_data = self.log_normalize(x)
        return self.net(norm_data)


class PlainDecoder(torch.nn.Module):
    """Decoder maps latent state to the actual observations
    We need to also account for the normalization done to the data.

    We constraint the decoder to output positive definite output

    """
    def __init__(self, latent_dim, nhidden, output_size, data_mean, data_std,
                 nlayers=3, activation_fn='Tanh', **kwargs):
        super(PlainDecoder, self).__init__()

        self.net = MLP(latent_dim, nhidden, output_size, nlayers=nlayers, activation_fn = activation_fn)

        self.register_buffer('mean', torch.from_numpy(data_mean))
        self.register_buffer('std',  torch.from_numpy(data_std))
        #self.mean = torch.from_numpy(data_mean)
        #self.std  = torch.from_numpy(data_std)

    def log_normalize(self,x):
        return (torch.log10(x) - self.mean) / self.std

    def undo_log_normalize(self,x):
        return torch.pow(10, x* self.std + self.mean)

    def forward(self, z, x0=None, z0=None):
        return self.undo_log_normalize(self.net(z))

class Decoder(torch.nn.Module):
    """Decoder maps latent state to the actual observations
    We need to also account for the normalization done to the data.

    We constraint the decoder to output positive definite output

    - catch here

    """
    def __init__(self, latent_dim, nhidden, output_size, data_log_mean,
                 data_log_std, nlayers=3, activation_fn='Tanh', data_mean=None,
                 data_std=None, data_mask=None, map_HI=None,
                 ):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.nhidden     = nhidden

        self.net = MLP(latent_dim, nhidden, output_size, nlayers=nlayers, activation_fn = activation_fn)

        self.register_buffer('mean', torch.from_numpy(data_log_mean))
        self.register_buffer('std',  torch.from_numpy(data_log_std))
        #self.mean = torch.from_numpy(data_log_mean)
        #self.std  = torch.from_numpy(data_log_std)

        # we have an additional way of parametrizing the "scale" over which the
        # decoder decodes the path
        # initially it is treated as a learnable parameter, but this factor in
        # reality varies from initial condtions to initial conditions
        self.scaling_factor = nn.Parameter(torch.zeros(output_size))

        if map_HI is not None:
            self.map_HI = map_HI
        else:
            self.map_HI = None


        # once data_mean is specified make sure data_std, and data_mask is also specified
        all_none     = data_mean is None and data_std is None and data_mask is None
        all_not_none = data_mean is not None and data_std is not None and data_mask is not None
        assert all_none or all_not_none, 'either you specify all of data_mean, data_std, data_mask, or just dont at all'

        # no log-normalize
        if data_mean is None:
            self.register_buffer('_mean', torch.zeros(output_size))
            self.register_buffer('_std', torch.ones(output_size))
            self.register_buffer('_mask', torch.zeros(output_size))
        else:
            self.register_buffer('_mean', torch.from_numpy(data_mean))
            self.register_buffer('_std', torch.from_numpy(data_std))
            self.register_buffer('_mask', torch.from_numpy(data_mask))
        clip_value = 0.1
        for p in self.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    def normalize(self, x):
        return (x - self._mean) / self._std

    def log_normalize(self,x):
        return (torch.log10(x) - self.mean) / self.std

    def undo_log_normalize(self,x):
        return torch.pow(10, x* self.std + self.mean)

    def forward(self, z, z0, x0):
        pos_output = self.net( torch.cat((z,z0,self.log_normalize(x0)),dim=-1 ))
        return self.map_to_abundance(pos_output, x0)

    def map_to_abundance(self, net_output, x0): #, scaling_factor=None,):
        #if scaling_factor is None:
        #    s_fac = self.scaling_factor.exp().view(1, -1)
        #else:
        #    s_fac = scaling_factor.exp().view(1, -1)

        s_fac = self.scaling_factor.exp().view(1,-1)

        exp_out = x0 * torch.exp(net_output * s_fac)
        # need to clip it such that it is bounded below from 0
        return exp_out

        if self.map_HI:
            if len(x0.shape) == 2:
                density = (x0[:,5]/0.24).unsqueeze(-1)
            else:
                density = (x0[:,:,5]/0.24).unsqueeze(-1)
            lin_out = density * 0.76 * (1 - torch.exp(-net_output.abs()*s_fac)+1e-10 )
        else:
            lin_out = torch.relu(x0 + net_output * s_fac) + 1e-10

        #TODO: ad-hoc way of enforcing positivity and (attempt to) reudcing the
        # fluctutation of HI

        #lin_out = 0.76*(1 - torch.exp( - net_output.abs() * s_fac ) +
        #                1e-10)
        return exp_out * (1 - self._mask) + lin_out* self._mask

class PlainAutoEncoder(torch.nn.Module):
    def __init__(self, latent_dim, input_dim, nhidden, log_data_mean, log_data_std,
                 nlayers=3, activation_fn='ELU', **kwargs):
        super().__init__()
        self.enc = Encoder(input_dim, nhidden, latent_dim, log_data_mean, log_data_std, nlayers=nlayers, activation_fn=activation_fn)
        self.dec = PlainDecoder(latent_dim, nhidden, input_dim, log_data_mean,
                                log_data_std, nlayers=nlayers,activation_fn = activation_fn)
        self.weights = torch.ones(10)

    def forward(self, x, x0=None):
        z = self.enc(x)
        xrecon = self.dec(z)
        return xrecon
    def loss_fn(self, x, x0=None):
        logxpred = self.enc.log_normalize(self(x))
        logx     = self.enc.log_normalize(x)
        return (self.weights*(logx-logxpred)/logx).abs().mean()

class PlainAutoEncoder(torch.nn.Module):
    def __init__(self, latent_dim, input_dim, nhidden, log_data_mean, log_data_std,
                 nlayers=3, activation_fn='ELU', **kwargs):
        super().__init__()
        self.enc = Encoder(input_dim, nhidden, latent_dim, log_data_mean, log_data_std, nlayers=nlayers, activation_fn=activation_fn)
        self.dec = PlainDecoder(latent_dim, nhidden, input_dim, log_data_mean,
                                log_data_std, nlayers=nlayers,activation_fn = activation_fn)
        self.weights = torch.ones(10)

    def forward(self, x, x0=None):
        z = self.enc(x)
        xrecon = self.dec(z)
        return xrecon
    def loss_fn(self, x, x0=None):
        logxpred = self.enc.log_normalize(self(x))
        logx     = self.enc.log_normalize(x)
        return (self.weights*(logx-logxpred)/logx).abs().mean()

class ICGuidedAutoEncoder(torch.nn.Module):
    def __init__(self, latent_dim, input_dim, nhidden, data_log_mean,
                 data_log_std, nlayers=3, activation_fn='ELU', data_mean=None,
                 data_std=None, data_mask=None, map_HI=None,
                 ):
        super().__init__()
        self.enc = Encoder(input_dim, nhidden, latent_dim, data_log_mean, data_log_std, nlayers=nlayers, activation_fn=activation_fn)
        self.dec = Decoder(latent_dim*2 + input_dim, nhidden, input_dim,
                           data_log_mean, data_log_std,
                           nlayers=nlayers,activation_fn = activation_fn,
                           data_mean=data_mean, data_std=data_std,
                           data_mask=data_mask, map_HI=map_HI,
                           )
        self.weights = torch.ones(10)
        self.map_HI = map_HI

        # no log-normalize
        if data_mean is None:
            self._mean = torch.zeros(input_dim)
            self._std  = torch.ones(input_dim)
            self._mask = torch.ones(input_dim)
        else:
            self._mean = torch.from_numpy(data_mean)
            self._std  = torch.from_numpy(data_std)
            self._mask = torch.from_numpy(data_mask)

    def forward(self, x, x0=None):
        ntime = x.shape[1]
        z = self.enc(x)
        if x0 is None:
            z0 = self.enc(x[:,0,:]).unsqueeze(1).repeat(1,ntime,1)
            x0 = x[:,0,:].unsqueeze(1).repeat(1,ntime,1)
        else:
            z0 = self.enc(x0)

        xrecon = self.dec(z,z0,x0)
        return xrecon

    def loss_fn(self, x):
        xpred = self(x)
        lin_loss = ((x - xpred) / x).abs()*self._mask
        #TODO: fixed this when we use this special mapping
        #if self.map_HI:
        #    lin_loss = ((x - xpred) / (0.76-x/density )).abs()*self._mask
        # we compute the log norm loss
        logxpred = self.enc.log_normalize(xpred)
        logx     = self.enc.log_normalize(x)
        log_loss = (logxpred - logx).abs()* (1 - self._mask)
        # and linear loss

        total_loss = lin_loss + log_loss
        #print(total_loss.mean((0,1)))

        return (total_loss).abs().mean()

class HierarchicalDecoder(Decoder):
    def __init__(self, latent_dim, nhidden, output_size, data_log_mean, data_log_std, nlayers=3, activation_fn='Tanh', data_mean=None, data_std=None, data_mask=None):
        super(HierarchicalDecoder, self).__init__(latent_dim, nhidden, output_size, data_log_mean, data_log_std, nlayers, activation_fn, data_mean, data_std, data_mask)
        self.MLPs = nn.ModuleList()
        self.sfactors = []
        # create a list of MLPs to use all possible combinations of element in latent dim
        for i in range(latent_dim):
            self.MLPs.append(MLP(2*(i+1)+output_size, nhidden, output_size, nlayers=nlayers, activation_fn =  activation_fn))
            self.sfactors.append(nn.Parameter(torch.zeros(output_size)))

    def forward(self, z, z0, x0):
        output_list = []
        xshape = x0.shape
        z, z0 = z.reshape(-1, self.latent_dim), z0.reshape(-1, self.latent_dim)
        x0 = x0.reshape(-1, self.output_size)
        for i, mlp in enumerate(self.MLPs):
            input_vec = torch.cat((z[:,:i+1],z0[:,:i+1],self.log_normalize(x0)),dim=-1 )
            output_list.append(self.map_to_abundance(mlp(input_vec), x0, self.sfactors[i]).reshape(*xshape))
        return output_list


class ICGuided_HierarchicalAutoEncoder(torch.nn.Module):
    def __init__(self, latent_dim, input_dim, nhidden, data_log_mean,
                 data_log_std, nlayers=3, activation_fn='ELU', data_mean=None,
                 data_std=None, data_mask=None, map_HI=None):
        super().__init__()
        self.enc = Encoder(input_dim, nhidden, latent_dim, data_log_mean, data_log_std, nlayers=nlayers, activation_fn=activation_fn)
        self.dec = HierarchicalDecoder(latent_dim, nhidden, input_dim, data_log_mean, data_log_std, nlayers=nlayers,activation_fn = activation_fn,
                                       data_mean=data_mean, data_std=data_std, data_mask=data_mask)

        # no log-normalize
        if data_mean is None:
            self._mean = torch.zeros(input_dim)
            self._std  = torch.ones(input_dim)
            self._mask = torch.ones(input_dim)
        else:
            self._mean = torch.from_numpy(data_mean)
            self._std  = torch.from_numpy(data_std)
            self._mask = torch.from_numpy(data_mask)

    def forward(self, x, x0=None):
        ntime = x.shape[1]
        z = self.enc(x)
        if x0 is None:
            z0 = self.enc(x[:,0,:]).unsqueeze(1).repeat(1,ntime,1)
            x0 = x[:,0,:].unsqueeze(1).repeat(1,ntime,1)
        else:
            z0 = self.enc(x0)

        xrecon = self.dec(z,z0,x0)
        return xrecon

    def _loss_fn(self, x, xpred):
        lin_loss = ((x - xpred) / x).abs()*self._mask
        # we compute the log norm loss
        logxpred = self.enc.log_normalize(xpred)
        logx     = self.enc.log_normalize(x)
        log_loss = (logxpred - logx).abs()* (1 - self._mask)
        # and linear loss

        total_loss = lin_loss + log_loss
        #print(total_loss.mean((0,1)))
        return (total_loss).abs().mean()

    def loss_fn(self, x):
        total_loss = 0
        xpred = self(x)
        for i in xpred:
            total_loss += self._loss_fn(x,i)
        return total_loss


import torch.functional as F
def sparse_loss(rho, images):
    values = images
    loss = 0
    for i in range(len(model_children)):
        if model_children[1].__module__ == "torch.nn.modules.activation":
            values = model_children[i](values)
            loss += kl_divergence(rho, values)
            print(kl_divergence(rho, values))

    return loss

def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), (1,2) ) # sigmoid because we need the probability distributions
    rho = torch.tensor([rho] * len(rho_hat))
    return torch.sum(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))
