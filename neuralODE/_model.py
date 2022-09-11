from neuralODE.utils import plot_example, prepare_dataloaders, get_default_hyperparams
from neuralODE.autoencoder import *
from neuralODE.model import ODEFunc
from torchdiffeq import odeint

class NormedNeuralODE(nn.Module):
    def __init__(self, rhs_func, tmult):
        super().__init__()
        self.model = rhs_func
        self.tmult = tmult
    def forward(self,t,x):
        return self.tmult * self.model(t,x)

class ScaledDecoder(Decoder):
    def __init__(self, *args, **kwargs):
        # unlike other Decoders, we also need a log mean and log std for the
        # actual scale factor, we have to pop it out before super init
        scale_log_mean = kwargs.pop("scale_log_mean")
        scale_log_std  = kwargs.pop("scale_log_std")

        super().__init__(*args, **kwargs)

        self.register_buffer("scale_log_mean", scale_log_mean)
        self.register_buffer("scale_log_std",  scale_log_std)

        for k, v in kwargs.items():
            setattr(self, k, v)

        # for scaling factor
        self.scale_net = MLP(self.output_size,
                             self.nhidden,
                             self.output_size,
                             nlayers=self.nlayers,
                             activation_fn = self.activation_fn)

    def map_to_abundance(self, net_output, x0): #, scaling_factor=None,):
        norm_x0 = self.log_normalize(x0)
        s_fac = self.scale_net(norm_x0)
        s_fac = self.scale_transform(s_fac).detach()
        exp_out = x0 * torch.exp(net_output * s_fac)
        return exp_out

    def scale_transform(self, s_fac):
        unnorm = s_fac* self.scale_log_std
        unnorm = unnorm + self.scale_log_mean
        return unnorm

class ScaledLatentODEfunc(nn.Module):
    def __init__(self,
                 log_mean=[],
                 log_std=[],
                 scale_log_mean=[],
                 scale_log_std=[],
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
                 ae_type = 'ICAE',
                 map_HI = None,
                 nlayers_ode=None,
                 ):
        super().__init__()

        AE= {"ICAE":    ICGuidedAutoEncoder,
             'PlainAE': PlainAutoEncoder,
             'ICHAE':   ICGuided_HierarchicalAutoEncoder}
        assert ae_type in AE

        self.autoencoder = AE[ae_type](latent_dim, nspecies,
                                        nhidden,
                                        log_mean, log_std,
                                        nlayers=nlayers,
                                        activation_fn=activation_fn,
                                        data_mean=data_mean,
                                        data_std=data_std,
                                        data_mask=data_mask,
                                        map_HI=map_HI,
                                        )

        # The idea is that we can parametrized an additional MLP
        # to learn the expected variation scale from the mapping
        # we can override the `map_to_abundance` part of the decoder
        # the scaling factor is instead an MLP that takes normalized x0
        # as inputs,

        self.autoencoder.dec = ScaledDecoder(
            latent_dim*2 + nspecies, nhidden, nspecies,
            log_mean, log_std,
            nlayers=nlayers,activation_fn = activation_fn,
            data_mean=data_mean, data_std=data_std,
            data_mask=data_mask, map_HI=map_HI,
            scale_log_mean=torch.from_numpy(scale_log_mean),
            scale_log_std =torch.from_numpy(scale_log_std),)


        if nlayers_ode is None:
            nlayers_ode = nlayers


        self.rhs_func = ODEFunc(latent_dim,
                                nhidden, nlayers=nlayers_ode,
                                activation_fn=activation_fn,
                                final_activation =ode_final_activation)

        self.nfe = 0

        self.mixing =mixing
        self.nspecies = nspecies
        self.latent_dim = latent_dim

        self.log_mean = torch.from_numpy(log_mean)
        self.log_std  = torch.from_numpy(log_std)
        self.map_HI = map_HI

    def encode_initial_condition(self, x0):
        return self.autoencoder.enc(x0)

    def integrate_ODE(self, t, z0):
        return odeint(self.rhs_func, z0, t)

    def decode_path(self, z_path, z0, x0):
        ntimesteps = z_path.shape[0]
        X0_repeat = x0.unsqueeze(1).repeat(1,ntimesteps,1).permute(1,0,2)
        z0_repeat = z0.unsqueeze(1).repeat(1,ntimesteps,1).permute(1,0,2)

        # decode a normalize value!
        output_dec = self.autoencoder.dec(z_path, z0_repeat, X0_repeat)

        if isinstance(output_dec, list):
            output_dec = output_dec[-1]
        x_pred     = output_dec.permute(1,2,0)

        return x_pred

    def reconstruction_loss(self, x, x0, z0):
        ntimesteps = x.shape[2]
        X0_repeat = x0.unsqueeze(2).repeat(1,1,ntimesteps).permute(0,2,1)
        z0_repeat = z0.unsqueeze(2).repeat(1,1,ntimesteps).permute(0,2,1)

        reconstruct_x = self.autoencoder(x.permute(0,2,1))
        real_x     = self.autoencoder.enc.log_normalize(x.permute(0,2,1))

        if isinstance(reconstruct_x, list):
            loss_recon = 0.0
            for recon_x in reconstruct_x:
                loss_recon += ((self.autoencoder.enc.log_normalize(recon_x) -
                                real_x)).abs().mean() / len(reconstruct_x)
        else:
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


    def forward_tff_batch(self, t_tff, x, tmult, weighted_loss=False):
        '''
        Args:
            t_tff: time axis normalized by its own freefall time
            x: abundance
            tmult: multiplication factor to normalize the RHS function
        return:
            x_pred: prediction
            loss_path:
            loss_recon: reconstruction loss
        '''
        assert ((t_tff / tmult).std(0) < 1e-10).all()

        x0 = x[:,:,0]
        z0     = self.encode_initial_condition(x0)

        # this is the part thats different from the self.forward
        #z_path = self.integrate_ODE(t[0], z0)

        normed_ode = NormedNeuralODE(self.rhs_func, tmult)
        z_path     = odeint(normed_ode, z0, t_tff[0]/tmult[0])

        x_pred = self.decode_path(z_path, z0, x0)
        real_x = self.autoencoder.enc.log_normalize(x.reshape(-1,self.nspecies))
        pred_x = self.autoencoder.enc.log_normalize(x_pred.reshape(-1,self.nspecies))

        # calculate weight according
        if weighted_loss:
            tidx = (t_tff <= 1).sum(axis=-1) - 1
            X_tff = x[torch.arange(len(tidx)),:,tidx]
            X_0   = x[:,:,0]
            pct_diff = torch.max((X_tff/X_0-1).abs(), torch.tensor([1e-6]))
            weight   = (1/pct_diff).unsqueeze(-1)

            # now the loss is the weighted percentage loss
            loss_path = (weight*(x - x_pred)/x).abs().mean()
            # reconstruction loss
            Xperm   = x.permute(0,2,1)
            _weight = weight.permute(0,2,1)
            Xrecon  = self.autoencoder(Xperm)
            loss_recon = (_weight*(Xperm-Xrecon) / Xperm).abs().mean()
        else:
            loss_path = ((pred_x -real_x)).abs().mean()
            loss_recon = self.reconstruction_loss(x, x0, z0)
        return x_pred, loss_path, loss_recon


    def forward_time_batch(self, t, x0, X_truth, obs_indx, taxis_indx,
                           get_pct_diff=None):
        z0     = self.encode_initial_condition(x0)
        z_path = self.integrate_ODE(t, z0)
        x_pred = self.decode_path(z_path, z0, x0)
        xpred_sort = x_pred[obs_indx,:, taxis_indx]

        # optimize the loss in the log-normalize space
        real_x = self.autoencoder.enc.log_normalize(X_truth)
        pred_x = self.autoencoder.enc.log_normalize(xpred_sort)
        loss_path = ((pred_x - real_x)).abs().mean()

        if get_pct_diff:
            real_x = torch.log(X_truth)
            pred_x = torch.log(xpred_sort)
            loss_path = ((pred_x - real_x)/ real_x).abs().mean()

        # additional loss that weights it by the deviation from unity
        x0_ = x0[obs_indx]
        xtruth_frac =  X_truth/x0_
        xpred_frac  =  xpred_sort/x0_
        loss_recon_unity = ((xtruth_frac - xpred_frac) / (1 -xtruth_frac+1e-6)).abs().mean()*1e-6
        loss_path += loss_recon_unity


        # now reconstruction loss
        reconstruct_x = self.autoencoder(X_truth, x0= x0[obs_indx])
        if isinstance(reconstruct_x, list):
            # case where the autoencoder is Hierachical
            loss_recon = 0.0
            for recon_x in reconstruct_x:
                loss_recon += ((self.autoencoder.enc.log_normalize(recon_x) -
                                real_x)).abs().mean() / len(reconstruct_x)
        else:
            loss_recon =  ((self.autoencoder.enc.log_normalize(reconstruct_x) - real_x)).abs().mean()

        if get_pct_diff:
            loss_recon =  ((torch.log(reconstruct_x) - real_x)/ real_x).abs().mean()

        return xpred_sort, loss_path, loss_recon




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
                 ae_type = 'ICAE',
                 map_HI = None,
                 nlayers_ode=None,
                 ):
        super(NewLatentODEfunc, self).__init__()

        AE= {"ICAE":    ICGuidedAutoEncoder,
             'PlainAE': PlainAutoEncoder,
             'ICHAE':   ICGuided_HierarchicalAutoEncoder}
        assert ae_type in AE

        self.autoencoder = AE[ae_type](latent_dim, nspecies,
                                        nhidden,
                                        log_mean, log_std,
                                        nlayers=nlayers,
                                        activation_fn=activation_fn,
                                        data_mean=data_mean,
                                        data_std=data_std,
                                        data_mask=data_mask,
                                        map_HI=map_HI,
                                        )

        if nlayers_ode is None:
            nlayers_ode = nlayers


        self.rhs_func = ODEFunc(latent_dim,
                                nhidden, nlayers=nlayers_ode,
                                activation_fn=activation_fn,
                                final_activation =ode_final_activation)

        self.nfe = 0

        self.mixing =mixing
        self.nspecies = nspecies
        self.latent_dim = latent_dim

        self.log_mean = torch.from_numpy(log_mean)
        self.log_std  = torch.from_numpy(log_std)
        self.map_HI = map_HI

    def encode_initial_condition(self, x0):
        return self.autoencoder.enc(x0)

    def integrate_ODE(self, t, z0):
        return odeint(self.rhs_func, z0, t)

    def decode_path(self, z_path, z0, x0):
        ntimesteps = z_path.shape[0]
        X0_repeat = x0.unsqueeze(1).repeat(1,ntimesteps,1).permute(1,0,2)
        z0_repeat = z0.unsqueeze(1).repeat(1,ntimesteps,1).permute(1,0,2)

        # decode a normalize value!
        output_dec = self.autoencoder.dec(z_path, z0_repeat, X0_repeat)

        if isinstance(output_dec, list):
            output_dec = output_dec[-1]
        x_pred     = output_dec.permute(1,2,0)

        return x_pred

    def reconstruction_loss(self, x, x0, z0):
        ntimesteps = x.shape[2]
        X0_repeat = x0.unsqueeze(2).repeat(1,1,ntimesteps).permute(0,2,1)
        z0_repeat = z0.unsqueeze(2).repeat(1,1,ntimesteps).permute(0,2,1)

        reconstruct_x = self.autoencoder(x.permute(0,2,1))
        real_x     = self.autoencoder.enc.log_normalize(x.permute(0,2,1))

        if isinstance(reconstruct_x, list):
            loss_recon = 0.0
            for recon_x in reconstruct_x:
                loss_recon += ((self.autoencoder.enc.log_normalize(recon_x) -
                                real_x)).abs().mean() / len(reconstruct_x)
        else:
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

    def forward_tff_batch(self, t_tff, x, tmult, weighted_loss=False):
        '''
        Args:
            t_tff: time axis normalized by its own freefall time
            x: abundance
            tmult: multiplication factor to normalize the RHS function
        return:
            x_pred: prediction
            loss_path:
            loss_recon: reconstruction loss
        '''
        assert ((t_tff / tmult).std(0) < 1e-10).all()

        x0 = x[:,:,0]
        z0     = self.encode_initial_condition(x0)

        # this is the part thats different from the self.forward
        #z_path = self.integrate_ODE(t[0], z0)

        normed_ode = NormedNeuralODE(self.rhs_func, tmult)
        z_path     = odeint(normed_ode, z0, t_tff[0]/tmult[0])

        x_pred = self.decode_path(z_path, z0, x0)

        # calculate weight according
        if weighted_loss:
            tidx = (t_tff <= 1).sum(axis=-1) - 1
            X_tff = x[torch.arange(len(tidx)),:,tidx]
            X_0   = x[:,:,0]
            pct_diff = torch.max((X_tff/X_0-1).abs(), torch.tensor([1e-2]))
            weight   = (1/pct_diff).unsqueeze(-1)
            weight = torch.max(weight, torch.tensor([1.0]))
            weight = torch.ones_like(weight)

            # reconstruction loss
            Xperm   = x.permute(0,2,1)
            _weight = weight.permute(0,2,1)
            Xrecon  = self.autoencoder(Xperm)
            #loss_recon = (_weight*(Xperm-Xrecon) / Xperm).abs().mean()
            loss_recon = (_weight*Xperm/Xrecon).log().abs().mean()

            # now the loss is the weighted percentage loss
            weight = torch.ones_like(x)
            #loss_path = (weight*(x - x_pred)/x).abs().mean()
            loss_path = (x/x_pred).log().abs().mean()

            # conservation loss
            # total hydrogen mass
            Xinit = x[:,:5,0].sum(axis=1).unsqueeze(-1)
            Xmass = x_pred[:,:5,:].sum(axis=1)
            #loss_cons = (((Xinit - Xmass ) / Xinit)).abs().mean()
            loss_cons = (Xinit/Xmass).log().abs().mean()
            loss_recon += loss_cons

        else:
            real_x = self.autoencoder.enc.log_normalize(x.reshape(-1,self.nspecies))
            pred_x = self.autoencoder.enc.log_normalize(x_pred.reshape(-1,self.nspecies))
            loss_path = ((pred_x -real_x)).abs().mean()
            loss_recon = self.reconstruction_loss(x, x0, z0)
        return x_pred, loss_path, loss_recon


    def forward_time_batch(self, t, x0, X_truth, obs_indx, taxis_indx,
                           get_pct_diff=None):
        z0     = self.encode_initial_condition(x0)
        z_path = self.integrate_ODE(t, z0)
        x_pred = self.decode_path(z_path, z0, x0)
        xpred_sort = x_pred[obs_indx,:, taxis_indx]

        # optimize the loss in the log-normalize space
        real_x = self.autoencoder.enc.log_normalize(X_truth)
        pred_x = self.autoencoder.enc.log_normalize(xpred_sort)
        loss_path = ((pred_x - real_x)).abs().mean()

        if get_pct_diff:
            real_x = torch.log(X_truth)
            pred_x = torch.log(xpred_sort)
            loss_path = ((pred_x - real_x)/ real_x).abs().mean()

        # additional loss that weights it by the deviation from unity
        x0_ = x0[obs_indx]
        xtruth_frac =  X_truth/x0_
        xpred_frac  =  xpred_sort/x0_
        loss_recon_unity = ((xtruth_frac - xpred_frac) / (1 -xtruth_frac+1e-6)).abs().mean()*1e-6
        loss_path += loss_recon_unity


        # now reconstruction loss
        reconstruct_x = self.autoencoder(X_truth, x0= x0[obs_indx])
        if isinstance(reconstruct_x, list):
            # case where the autoencoder is Hierachical
            loss_recon = 0.0
            for recon_x in reconstruct_x:
                loss_recon += ((self.autoencoder.enc.log_normalize(recon_x) -
                                real_x)).abs().mean() / len(reconstruct_x)
        else:
            loss_recon =  ((self.autoencoder.enc.log_normalize(reconstruct_x) - real_x)).abs().mean()

        if get_pct_diff:
            loss_recon =  ((torch.log(reconstruct_x) - real_x)/ real_x).abs().mean()

        return xpred_sort, loss_path, loss_recon




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
