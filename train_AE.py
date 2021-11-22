import torch
from train import prepare_dataset
import time
from neuralODE.model import PlainDecoder, MLP
import numpy as np

import torch.nn as nn

class Decoder(torch.nn.Module):
    """Decoder maps latent state to the actual observations
    We need to also account for the normalization done to the data.

    We constraint the decoder to output positive definite output

    - catch here

    """
    def __init__(self, latent_dim, nhidden, output_size, data_log_mean, data_log_std, nlayers=3, activation_fn='Tanh', data_mean=None, data_std=None, data_mask=None):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size

        self.net = MLP(latent_dim, nhidden, output_size, nlayers=nlayers, activation_fn = activation_fn)

        self.mean = torch.from_numpy(data_log_mean)
        self.std  = torch.from_numpy(data_log_std)
        self.scaling_factor = nn.Parameter(torch.zeros(output_size))


        # once data_mean is specified make sure data_std, and data_mask is also specified
        all_none     = data_mean is None and data_std is None and data_mask is None
        all_not_none = data_mean is not None and data_std is not None and data_mask is not None
        assert all_none or all_not_none, 'either you specify all of data_mean, data_std, data_mask, or just dont at all'

        # no log-normalize
        if data_mean is None:
            self._mean = torch.zeros(output_size)
            self._std  = torch.ones(output_size)
            self._mask = torch.ones(output_size)
        else:
            self._mean = torch.from_numpy(data_mean)
            self._std  = torch.from_numpy(data_std)
            self._mask = torch.from_numpy(data_mask)

    def normalize(self, x):
        return (x - self._mean) / self._std

    def log_normalize(self,x):
        return (torch.log10(x) - self.mean) / self.std

    def undo_log_normalize(self,x):
        return torch.pow(10, x* self.std + self.mean)

    def forward(self, z, z0, x0):
        pos_output = self.net( torch.cat((z,z0,self.log_normalize(x0)),dim=-1 ))
        return self.map_to_abundance(pos_output, x0)

    def map_to_abundance(self, net_output, x0, scaling_factor=None):
        if scaling_factor is None:
            s_fac = self.scaling_factor.exp().view(1, -1)
        else:
            s_fac = scaling_factor.exp().view(1, -1)
        exp_out = x0 * torch.exp(net_output * s_fac)
        # need to clip it such that it is bounded below from 0
        lin_out = torch.relu(x0 + net_output * s_fac) + 1e-10
        return exp_out * (1 - self._mask) + lin_out* self._mask




def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), (1,2) ) # sigmoid because we need the probability distributions
    rho = torch.tensor([rho] * len(rho_hat))
    return torch.sum(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))


class Encoder(torch.nn.Module):
    def __init__(self, input_size, nhidden, latent_dim, data_mean, data_std, nlayers=3, activation_fn='Tanh'):
        super(Encoder, self).__init__()

        self.net = MLP(input_size, nhidden, latent_dim, nlayers=nlayers, activation_fn = activation_fn)
        self.mean = torch.from_numpy(data_mean)
        self.std  = torch.from_numpy(data_std)
        self.enc_children = list(self.net.net.children())

    def log_normalize(self,x):
        return (torch.log10(x) - self.mean) / self.std

    def forward(self, x):
        norm_data = self.log_normalize(x)
        return self.net(norm_data)

    def sparse_loss(self, x):
        loss = 0.0
        for i in range(len(self.enc_children)):
            if self.enc_children[i].__module__ == "torch.nn.modules.activation":
                values = self.children[i](values)
                loss += kl_divergence(rho, values)
        return loss



def train_step_multi_dataloaders(model, optimizer, dataloaders):
    total_loss = 0.0
    time0 = time.time()
    for data_tuple in zip(*dataloaders):
        optimizer.zero_grad()
        loss = 0.0
        for t, X in data_tuple:
            l= model.loss_fn(X.permute(0,2,1).double())
            loss += l
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item() / sum(map(len,dataloaders))
    train_time = time.time() - time0
    return total_loss, train_time

def val_step_multi_dataloaders(model, dataloaders):
    with torch.no_grad():
        total_loss = 0.0
        time0 = time.time()
        for data_tuple in zip(*dataloaders):
            loss = 0.0
            for t, X in data_tuple:
                l= model.loss_fn(X.permute(0,2,1).double())
                loss += l
            total_loss += loss.detach().item() / sum(map(len,dataloaders))
    train_time = time.time() - time0
    return total_loss, train_time

def plot_example(model, t, X):
    sp_names = ['H2I', 'H2II', 'HI', 'HII', 'HM', 'HeI', 'HeII', 'HeIII', 'de', 'ge']
    f,axes = plt.subplots(2,10//2,figsize=(20,10))
    with torch.no_grad():
        X_recon = model(X.permute(0,2,1).double())
        for i in range(10):
            for j in range(6):
                axes.flat[i].loglog(t[j], X[j,i,:].detach(), c = f"C{j}")
                axes.flat[i].plot(t[j], X_recon[j,:,i].detach(), marker='.', c = f"C{j%6}", alpha=0.5)

                axes.flat[i].set_title(sp_names[i])
                plt.tight_layout()

class PlainAutoEncoder(torch.nn.Module):
    def __init__(self, latent_dim, input_dim, nhidden, data_mean, data_std, nlayers=3, activation_fn='ELU'):
        super().__init__()
        self.enc = Encoder(input_dim, nhidden, latent_dim, data_mean, data_std, nlayers=nlayers, activation_fn=activation_fn)
        self.dec = PlainDecoder(latent_dim, nhidden, input_dim, data_mean, data_std, nlayers=nlayers,activation_fn = activation_fn)

    def forward(self, x):
        z = self.enc(x)
        xrecon = self.dec(z)
        return xrecon
    def loss_fn(self, x):
        logxpred = self.enc.log_normalize(self(x))
        logx     = self.enc.log_normalize(x)
        return ((logx-logxpred)).abs().mean()

class ICGuidedAutoEncoder(torch.nn.Module):
    def __init__(self, latent_dim, input_dim, nhidden, data_log_mean, data_log_std, nlayers=3, activation_fn='ELU', data_mean=None, data_std=None, data_mask=None):
        super().__init__()
        self.enc = Encoder(input_dim, nhidden, latent_dim, data_log_mean, data_log_std, nlayers=nlayers, activation_fn=activation_fn)
        self.dec = Decoder(latent_dim*2 + input_dim, nhidden, input_dim, data_log_mean, data_log_std, nlayers=nlayers,activation_fn = activation_fn, data_mean=data_mean, data_std=data_std, data_mask=data_mask)
        self.weights = torch.ones(10)

        # no log-normalize
        if data_mean is None:
            self._mean = torch.zeros(input_size)
            self._std  = torch.ones(input_size)
            self._mask = torch.ones(input_size)
        else:
            self._mean = torch.from_numpy(data_mean)
            self._std  = torch.from_numpy(data_std)
            self._mask = torch.from_numpy(data_mask)

    def forward(self, x):
        ntime = x.shape[1]
        z = self.enc(x)
        z0 = self.enc(x[:,0,:]).unsqueeze(1).repeat(1,ntime,1)
        x0 = x[:,0,:].unsqueeze(1).repeat(1,ntime,1)

        xrecon = self.dec(z,z0,x0)
        return xrecon

    def loss_fn(self, x):
        xpred = self(x)
        lin_loss = ((x - xpred) / x).abs()*self._mask
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
    def __init__(self, latent_dim, input_dim, nhidden, data_log_mean, data_log_std, nlayers=3, activation_fn='ELU', data_mean=None, data_std=None, data_mask=None):
        super().__init__()
        self.enc = Encoder(input_dim, nhidden, latent_dim, data_log_mean, data_log_std, nlayers=nlayers, activation_fn=activation_fn)
        self.dec = HierarchicalDecoder(latent_dim, nhidden, input_dim, data_log_mean, data_log_std, nlayers=nlayers,activation_fn = activation_fn,
                                       data_mean=data_mean, data_std=data_std, data_mask=data_mask)

        # no log-normalize
        if data_mean is None:
            self._mean = torch.zeros(input_size)
            self._std  = torch.ones(input_size)
            self._mask = torch.ones(input_size)
        else:
            self._mean = torch.from_numpy(data_mean)
            self._std  = torch.from_numpy(data_std)
            self._mask = torch.from_numpy(data_mask)

    def forward(self, x):
        ntime = x.shape[1]
        z = self.enc(x)
        z0 = self.enc(x[:,0,:]).unsqueeze(1).repeat(1,ntime,1)
        x0 = x[:,0,:].unsqueeze(1).repeat(1,ntime,1)

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

def train_not_norm_abundance():
    hyperparams = {}

# dataloader related
    hyperparams['batch_size']       = 32
    hyperparams['train_proportion'] = 0.01
    hyperparams['nworkers'] = 40
    hyperparams['batch_size'] = 32

# model related
    hyperparams['nspecies']   = 10
    hyperparams['latent_dim'] = 3
    hyperparams['nhidden']    = 32
    hyperparams['nlayers']    = 4
    hyperparams['activation_fn']     = 'ELU'
    hyperparams['use_plain_decoder'] = False
    hyperparams['initial_lr'] = 1e-3
    hyperparams['min_lr']     = 1e-6

# train related
    hyperparams['max_epochs']      = 1000
    hyperparams['t_mult']          = 1e-13
    hyperparams['lookback_period'] = 20
    hyperparams['run_avg_beta']    = 0.99

    dl_train5, dl_test5, dl_val5 = prepare_dataset(dataname='new_dd0053_chemistry_5.hdf5',
                                                   train_proportion=hyperparams['train_proportion'],
                                                   normalize_abundance=False,
                                                   batch_size=hyperparams['batch_size'],
                                                   num_workers=hyperparams['nworkers'])
    dl_train4, dl_test4, dl_val4 = prepare_dataset(dataname='new_dd0053_chemistry_4.hdf5',
                                                   train_proportion=hyperparams['train_proportion'],
                                                   normalize_abundance=False,
                                                   #sample_score_file='ds4_score.pt',
                                                   batch_size=hyperparams['batch_size'],
                                                   num_workers=hyperparams['nworkers'])
    dl_train3, dl_test3, dl_val3 = prepare_dataset(dataname='new_dd0053_chemistry_3.hdf5',
                                                   train_proportion=hyperparams['train_proportion'],
                                                   normalize_abundance=False,
                                                   #sample_score_file='ds3_score.pt',
                                                   batch_size=hyperparams['batch_size'],
                                                   num_workers=hyperparams['nworkers'])

    dl_train = [dl_train3, dl_train4, dl_train5]
    dl_test  = [dl_test3,  dl_test4,  dl_test5 ]
    dl_val   = [dl_val3,   dl_val4,   dl_val5  ]

    log_mean = np.array([-19.5817, -32.2001, -16.9687, -24.0315, -30.7553, -17.4677, -25.8998,
             -35.0746, -24.0296,  10.8056], dtype=np.float32)

    log_std = np.array([2.1041, 1.4316, 1.8485, 1.0977, 1.3115, 1.8497, 1.9498, 2.0691, 1.1023,
                                0.1916], dtype=np.float32)


    model_plainAE = PlainAutoEncoder(3, 10, 64, log_mean, log_std, nlayers=4, activation_fn='ELU').double()

    optimizer = torch.optim.Adam(model_plainAE.parameters())

    train_history = []
    val_history   = []
    for s in range(100):
        train_loss, train_time = train_step_multi_dataloaders(model_plainAE, optimizer, dl_train)
        val_loss, val_time     = val_step_multi_dataloaders(model_plainAE, dl_val)
        train_history.append(train_loss)
        val_history.append(val_loss)
        print(s, train_loss, train_time)
    torch.save(model_plainAE, "plain_AE_unnorm_64.pt")
    loss = {}
    loss['train_loss'] = train_history
    loss['val_loss']   = val_history

    torch.save(loss, "plain_AE_loss_unnorm_64.pt")
    print('done')


if __name__ == "__main__":
    train_not_norm_abundance()

    hyperparams = {}

# dataloader related
    hyperparams['batch_size']       = 32
    hyperparams['train_proportion'] = 0.01
    hyperparams['nworkers'] = 40
    hyperparams['batch_size'] = 32

# model related
    hyperparams['nspecies']   = 10
    hyperparams['latent_dim'] = 3
    hyperparams['nhidden']    = 32
    hyperparams['nlayers']    = 4
    hyperparams['activation_fn']     = 'ELU'
    hyperparams['use_plain_decoder'] = False
    hyperparams['initial_lr'] = 1e-3
    hyperparams['min_lr']     = 1e-6

# train related
    hyperparams['max_epochs']      = 1000
    hyperparams['t_mult']          = 1e-13
    hyperparams['lookback_period'] = 20
    hyperparams['run_avg_beta']    = 0.99

    dl_train5, dl_test5, dl_val5 = prepare_dataset(dataname='new_dd0053_chemistry_5.hdf5',
                                                   train_proportion=hyperparams['train_proportion'],
                                                   #sample_score_file='ds5_score.pt',
                                                   batch_size=hyperparams['batch_size'],
                                                   num_workers=hyperparams['nworkers'])
    dl_train4, dl_test4, dl_val4 = prepare_dataset(dataname='new_dd0053_chemistry_4.hdf5',
                                                   train_proportion=hyperparams['train_proportion'],
                                                   #sample_score_file='ds4_score.pt',
                                                   batch_size=hyperparams['batch_size'],
                                                   num_workers=hyperparams['nworkers'])
    dl_train3, dl_test3, dl_val3 = prepare_dataset(dataname='new_dd0053_chemistry_3.hdf5',
                                                   train_proportion=hyperparams['train_proportion'],
                                                   #sample_score_file='ds3_score.pt',
                                                   batch_size=hyperparams['batch_size'],
                                                   num_workers=hyperparams['nworkers'])

    dl_train = [dl_train3, dl_train4, dl_train5]
    dl_test  = [dl_test3,  dl_test4,  dl_test5 ]
    dl_val   = [dl_val3,   dl_val4,   dl_val5  ]

    log_mean = np.array([ -2.7338, -15.3522,  -0.1207,  -7.1836, -13.9073, -17.4677,  -9.0518,
                         -18.2267,  -7.1817,  10.8056], dtype=np.float32)

    log_std  = np.array([0.2888, 0.5356, 0.0056, 0.7999, 0.6158, 1.8497, 0.1048, 0.2287, 0.7949,
                         0.1916], dtype=np.float32)


    # normalization method for HI
    mean = torch.zeros(10).numpy()
    mean[2] = 0.76
    std = torch.zeros(10).numpy()
    std[2] = 0.0002
    mask  = torch.zeros(10).numpy()
    mask[2] = 1.0


    model_plainAE = PlainAutoEncoder(3, 10, 64, log_mean, log_std, nlayers=4, activation_fn='ELU').double()

    optimizer = torch.optim.Adam(model_plainAE.parameters())

    train_history = []
    val_history   = []
    for s in range(100):
        train_loss, train_time = train_step_multi_dataloaders(model_plainAE, optimizer, dl_train)
        val_loss, val_time     = val_step_multi_dataloaders(model_plainAE, dl_val)
        train_history.append(train_loss)
        val_history.append(val_loss)
        print(s, train_loss, train_time)
    torch.save(model_plainAE, "plain_AE_64.pt")
    loss = {}
    loss['train_loss'] = train_history
    loss['val_loss']   = val_history

    torch.save(loss, "plain_AE_loss_64.pt")
    print('done')

    model_ICHAE = ICGuided_HierarchicalAutoEncoder(3, 10, 64, log_mean, log_std,  nlayers=4, activation_fn='ELU', data_mean=mean, data_std=std, data_mask=mask).double()
    optimizer = torch.optim.Adam(model_ICHAE.parameters())

    train_history = []
    val_history   = []
    for s in range(100):
        train_loss, train_time = train_step_multi_dataloaders(model_ICHAE, optimizer, dl_train)
        val_loss, val_time     = val_step_multi_dataloaders(model_ICHAE, dl_val)
        train_history.append(train_loss)
        val_history.append(val_loss)
        print(s, train_loss, train_time)
    torch.save(model_ICHAE, "ICHAE_64.pt")
    loss = {}
    loss['train_loss'] = train_history
    loss['val_loss']   = val_history
    torch.save(loss, "ICHAE_loss_64.pt")

    model_ICAE = ICGuidedAutoEncoder(3, 10, 64, log_mean, log_std,  nlayers=4, activation_fn='ELU', data_mean=mean, data_std=std, data_mask=mask).double()
    optimizer = torch.optim.Adam(model_ICAE.parameters())

    train_history = []
    val_history   = []
    for s in range(100):
        train_loss, train_time = train_step_multi_dataloaders(model_ICAE, optimizer, dl_train)
        val_loss, val_time     = val_step_multi_dataloaders(model_ICAE, dl_val)
        train_history.append(train_loss)
        val_history.append(val_loss)
        print(s, train_loss, train_time)
    torch.save(model_ICAE, "IC_AE_64.pt")
    loss = {}
    loss['train_loss'] = train_history
    loss['val_loss']   = val_history

    torch.save(loss, "IC_AE_loss_64.pt")
