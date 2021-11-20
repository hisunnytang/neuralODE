from neuralODE.utils import plot_example, prepare_dataloaders, get_default_hyperparams
from neuralODE.autoencoder import *
from neuralODE.model import ODEFunc
from torchdiffeq import odeint

class NewLatentODEfunc(nn.Module):
    def __init__(self,
                 log_mean,
                 log_std,
                 latent_dim=3,
                 nlayers=4,
                 nhidden=20,
                 mixing=1.0,
                 nspecies=10,
                 use_binaryODE=False,
                 activation_fn = 'SiLU',
                 data_mean=None,
                 data_std=None,
                 data_mask=None,
                 ode_final_activation=None,
                 ae_type = 'ICAE'):
        super(NewLatentODEfunc, self).__init__()

        AE= {"ICAE":    ICGuidedAutoEncoder,
             'PlainAE': PlainAutoEncoder,
             'ICHAE':   ICGuided_HierarchicalAutoEncoder}
        assert ae_type in AE

        self.autoencoder = AE[ae_type](latent_dim, nspecies,
                                        nhidden,
                                        log_mean, log_std,
                                        nlayers,
                                        activation_fn,
                                        data_mean, data_std, data_mask)


        self.rhs_func = ODEFunc(latent_dim,
                                nhidden, nlayers=nlayers,
                                activation_fn=activation_fn,
                                final_activation =ode_final_activation)

        self.nfe = 0

        self.mixing =mixing
        self.nspecies = nspecies
        self.latent_dim = latent_dim

        self.log_mean = torch.from_numpy(log_mean)
        self.log_std  = torch.from_numpy(log_std)

    def encode_initial_condition(self, x0):
        return self.autoencoder.enc(x0)

    def integrate_ODE(self, t, z0):
        return odeint(self.rhs_func, z0, t)

    def decode_path(self, z_path, z0, x0):
        ntimesteps = z_path.shape[0]
        X0_repeat = x0.unsqueeze(1).repeat(1,ntimesteps,1).permute(1,0,2)
        z0_repeat = z0.unsqueeze(1).repeat(1,ntimesteps,1).permute(1,0,2)

        # decode a normalize value!
        x_pred = self.autoencoder.dec(z_path, z0_repeat, X0_repeat).permute(1,2,0)

        return x_pred

    def reconstruction_loss(self, x, x0, z0):
        ntimesteps = x.shape[2]
        X0_repeat = x0.unsqueeze(2).repeat(1,1,ntimesteps).permute(0,2,1)
        z0_repeat = z0.unsqueeze(2).repeat(1,1,ntimesteps).permute(0,2,1)

        reconstruct_x = self.autoencoder(x.permute(0,2,1))
        real_x     = self.autoencoder.enc.log_normalize(x.permute(0,2,1))
        loss_recon =  ((self.autoencoder.enc.log_normalize(reconstruct_x) - real_x)).abs().mean()
        return loss_recon

    def forward(self, t, x):
        x0 = x[:,:,0]
        z0     = self.encode_initial_condition(x0)
        z_path = self.integrate_ODE(t[0], z0)
        x_pred = self.decode_path(z_path, z0, x0)

        real_x = self.autoencoder.enc.log_normalize(x.reshape(-1,self.nspecies))
        pred_x = self.autoencoder.enc.log_normalize(x_pred.reshape(-1,self.nspecies))
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
