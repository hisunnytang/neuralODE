import os
import math
import time
import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.model_selection import train_test_split
import h5py

class H5ICDataset(torch.utils.data.Dataset):
    def __init__(self, filename, mean, std):
        self.filename = filename
        self.sp_names = self.get_species_name()
        self.length   = self.get_length()
        self.nspecies = len(self.sp_names)
        self.mean, self.std = mean, std
    
    def get_species_name(self):
        with h5py.File(self.filename, 'r') as f:
            sp_paths = f.attrs['data_columns']
        return list(sorted(sp_paths))
    
    def get_length(self):
        with h5py.File(self.filename, 'r') as f:
            data_length = f['data_group'].shape[0]
        return int(data_length)
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        X = np.zeros((self.nspecies))
        with h5py.File(self.filename, 'r') as f:
            X = f['data_group'][idx][:,0]
            density = X[5:6]/0.24
            X[:5] /= density
            X[6:-1] /= density            
            X = (torch.log10(torch.from_numpy(X)) - self.mean) / self.std
        return X.float()
    
    
class AbundFlow(pl.LightningModule):

    def __init__(self, flows, import_samples=8):
        """
        Inputs:
            flows - A list of flows (each a nn.Module) that should be applied on the images.
            import_samples - Number of importance samples to use during testing (see explanation below). Can be changed at any time
        """
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.import_samples = import_samples
        # Create prior distribution for final latent space
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        # Example input for visualizing the graph
#         self.example_input_array = train_set[0][0].unsqueeze(dim=0)

    def forward(self, imgs):
        # The forward function is only used for visualizing the graph
        return self._get_likelihood(imgs)

    def encode(self, imgs):
        # Given a batch of images, return the latent representation z and ldj of the transformations
        z, ldj = imgs, torch.zeros(imgs.shape[0])
        for flow in self.flows:
            z, ldj = flow(z, ldj, reverse=False)
        return z, ldj

    def _get_likelihood(self, imgs, return_ll=False):
        """
        Given a batch of images, return the likelihood of those.
        If return_ll is True, this function returns the log likelihood of the input.
        Otherwise, the ouptut metric is bits per dimension (scaled negative log likelihood)
        """
        z, ldj = self.encode(imgs)
        log_pz = self.prior.log_prob(z).sum(dim=[1])
        log_px = ldj + log_pz
        nll = -log_px
        # Calculating bits per dimension
        bpd = nll * np.log2(np.exp(1)) / np.prod(imgs.shape[1:])
        return bpd.mean() if not return_ll else log_px

    @torch.no_grad()
    def sample(self, img_shape, z_init=None):
        """
        Sample a batch of images from the flow.
        """
        # Sample latent representation from prior
        if z_init is None:
            z = self.prior.sample(sample_shape=img_shape)
        else:
            z = z_init

        # Transform z to x by inverting the flows
        ldj = torch.zeros(img_shape[0])
        for flow in reversed(self.flows):
            z, ldj = flow(z, ldj, reverse=True)
        return z

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        # An scheduler is optional, but can help in flows to get the last bpd improvement
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # Normalizing flows are trained by maximum likelihood => return bpd
        loss = self._get_likelihood(batch[0])
        self.log('train_bpd', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_likelihood(batch[0])
        self.log('val_bpd', loss)

class CouplingLayer(nn.Module):

    def __init__(self, network, mask, c_in):
        """
        Coupling layer inside a normalizing flow.
        Inputs:
            network - A PyTorch nn.Module constituting the deep neural network for mu and sigma.
                      Output shape should be twice the channel size as the input.
            mask - Binary mask (0 or 1) where 0 denotes that the element should be transformed,
                   while 1 means the latent will be used as input to the NN.
            c_in - Number of input channels
        """
        super().__init__()
        self.network = network
        self.scaling_factor = nn.Parameter(torch.zeros(c_in))
        # Register mask as buffer as it is a tensor which is not a parameter,
        # but should be part of the modules state.
        self.register_buffer('mask', mask)

    def forward(self, z, ldj, reverse=False):
        """
        Inputs:
            z - Latent input to the flow
            ldj - The current ldj of the previous flows.
                  The ldj of this layer will be added to this tensor.
            reverse - If True, we apply the inverse of the layer.
            orig_img (optional) - Only needed in VarDeq. Allows external
                                  input to condition the flow on (e.g. original image)
        """
        # Apply network to masked input
        z_in = z * self.mask
        nn_out = self.network(z_in)
        s, t = nn_out.chunk(2, dim=1)

        # Stabilize scaling output
        s_fac = self.scaling_factor.exp().view(1, -1)
        s = torch.tanh(s / s_fac) * s_fac

        # Mask outputs (only transform the second part)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z = (z + t) * torch.exp(s)
            ldj += s.sum(dim=[1])
        else:
            z = (z * torch.exp(-s)) - t
            ldj -= s.sum(dim=[1])

        return z, ldj
    
class MLP(nn.Module):
    def __init__(self, input_dim, nhidden, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, nhidden),
#             nn.ELU(),
#             nn.Linear(nhidden, nhidden),
            nn.ELU(),
            nn.Linear(nhidden, output_dim, bias=False),
        )
    def forward(self, data):
        return self.net(data)
    
def get_mask(idx):
    mask = torch.arange(10)
    mask1 = (mask > 4).int().unsqueeze(0)
    mask2 = (mask <= 4).int().unsqueeze(0)
    if idx %2 == 1:
        return mask1
    return mask2