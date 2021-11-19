import torch
from torchdiffeq import odeint

import h5py
import glob
import numpy as np
import time

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from numpy.random import default_rng

from neuralODE.dataloader import H5ODEDataset
from neuralODE.model import LatentODEfunc, LatentODEfunc_PlainDec


from torch.optim.lr_scheduler import ReduceLROnPlateau

import os

# tensorboard setup
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

RNG_SEED = 42

np.random.seed(RNG_SEED)

def prepare_dataset(dataname='new_dd0053_chemistry_5.hdf5', train_proportion=0.1, batch_size=128, num_workers=40, sample_score_file=None, normalize_abundance=True):
    assert 0 < train_proportion  <= 1, 'proportion of training data is bound betwen 0,1'

    ds = H5ODEDataset(dataname, normalize_abundance=normalize_abundance)

    nsamples = int( len(ds)* train_proportion )
    subset_indexes = np.random.choice(len(ds),nsamples)

    idx_train, idx_val_test = train_test_split(subset_indexes, test_size=0.2, random_state=RNG_SEED)
    idx_val  , idx_test     = train_test_split(idx_val_test, test_size=0.5, random_state=RNG_SEED)

    ds_train = Subset(ds, idx_train)
    ds_val   = Subset(ds, idx_val)
    ds_test  = Subset(ds, idx_test)

    # select the weights from idx
    # score is the output log p(x)
    if sample_score_file is not None:
        score = torch.load(sample_score_file)
        assert len(score) == len(ds), f"weight file {sample_weight_file} should have the same number of datapoints as dataset {h5filename}"
        train_score   = torch.from_numpy(score[idx_train])
        weights = torch.maximum( torch.exp(-0.01*train_score), torch.tensor([1e-3]) )
        train_sampler = WeightedRandomSampler(weights, len(idx_train))
        shuffle= False
    else:
        train_sampler = None
        shuffle = True


    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=train_sampler)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_val  = DataLoader(ds_val,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dl_train, dl_test, dl_val


def prepare_model(log_mean_list, log_std_list, nspecies=10, latent_dim=3, nlayers=4, nhidden=32, activation_fn='SiLU', ode_final_activation=None, use_plain_decoder = False):
    if use_plain_decoder:
        latent_ode_function = LatentODEfunc_PlainDec(log_mean_list,
                                            log_std_list,
                                            latent_dim=latent_dim,
                                            nspecies=nspecies,
                                            use_binaryODE=False,
                                            activation_fn=activation_fn,
                                            nlayers=nlayers,
                                            nhidden=nhidden,
                                            ode_final_activation=ode_final_activation
                                            ).double()
        return latent_ode_function
    latent_ode_function = LatentODEfunc(log_mean_list,
                                        log_std_list,
                                        latent_dim=latent_dim,
                                        nspecies=nspecies,
                                        use_binaryODE=False,
                                        activation_fn=activation_fn,
                                        nlayers=nlayers,
                                        nhidden=nhidden,
                                        ode_final_activation=ode_final_activation
                                        ).double()
    return latent_ode_function

def reconstruction_loss(model, x):
    x0 = x[:,:,0]
    z0 = model.encode_initial_condition(x0)
    ntimesteps = x.shape[2]
    X0_repeat = x0.unsqueeze(2).repeat(1,1,ntimesteps).permute(0,2,1)
    z0_repeat = z0.unsqueeze(2).repeat(1,1,ntimesteps).permute(0,2,1)

    z = model.encoder(x.permute(0,2,1))
    reconstruct_x = model.decoder(z, z0_repeat, X0_repeat)

    # full autoencoder loss
    real_x = model.encoder.log_normalize(x.permute(0,2,1))
    loss_recon =  ((model.encoder.log_normalize(reconstruct_x) - real_x)).abs().mean()
    return loss_recon


def train_step_multi_dataloaders(model, optimizer, dataloaders, t_mult):
    total_loss = 0.0
    time0 = time.time()
    for data_tuple in zip(*dataloaders):
        # the strategy is to step optimizer once we go through each dataloader once
        optimizer.zero_grad()
        loss = 0.0
        for t, X in data_tuple:
            abund_paths, loss_path, loss_recon = model(t* t_mult, X)
            loss += loss_path + loss_recon
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item() / sum(map(len,dataloaders))
    train_time = time.time() - time0
    return total_loss, train_time

def val_step_multi_dataloaders(model, dataloaders, t_mult):
    time0 = time.time()
    with torch.no_grad():
        total_loss = 0.0
        for data_tuple in zip(*dataloaders):
            for t, X in data_tuple:

                abund_paths, loss_path, loss_recon = model(t* t_mult, X)
                total_loss += (loss_path + loss_recon).item() / sum(map(len,dataloaders))
    val_time = time.time() - time0
    return total_loss, val_time

def train_step(model, optimizer, dataloaders, t_mult):
    if isinstance(dataloaders, list): return train_step_multi_dataloaders(model, optimizer, dataloaders, t_mult)
    total_loss = 0.0
    time0 = time.time()
    for t, X in dataloaders:

        optimizer.zero_grad()
        abund_paths, loss_path, loss_recon = model(t* t_mult,X)
        loss = loss_path + loss_recon
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item() / len(dataloaders)
    train_time = time.time() - time0
    return total_loss, train_time

def val_step(model, dataloaders, t_mult):
    if isinstance(dataloaders, list): return val_step_multi_dataloaders(model, dataloaders, t_mult)
    with torch.no_grad():
        total_loss = 0.0
        time0 = time.time()
        for t, X in dataloaders:
            abund_paths, loss_path, loss_recon = model(t* t_mult,X)
            loss = loss_path + loss_recon
            total_loss += loss.item() / len(dataloaders)
        val_time = time.time() - time0
    return total_loss, val_time


def train_loop(model, dataloaders, scheduler, writer, run_avg_beta = 0.99, max_epochs=10000, t_mult = 1e-11, checkpoint_version_dir = '.', lookback_period = 20):


    train_loss_history, val_loss_history     = [], []
    running_history,    running_val_history  = [], []

    β = run_avg_beta

    dl_train, dl_test, dl_val = dataloaders

    for i in range(max_epochs):
        # training and validation step
        total_loss, train_time = train_step(model, optimizer, dl_train, t_mult)
        total_val_loss, val_time = val_step(model, dl_test, t_mult)


        # keep track of the loss history
        train_loss_history.append(total_loss)
        val_loss_history.append(total_val_loss)

        # calculating running traning history
        if i == 0:
            running_history.append( total_loss/ (1-β) )
            running_val_history.append( total_val_loss/ (1-β) )
        else:
            running_history.append( running_history[-1]* β + total_loss* (1-β) )
            running_val_history.append( running_val_history[-1]* β + total_val_loss* (1-β))

        # add the loss to summary writer
        writer.add_scalar('training_loss', total_loss, i)
        writer.add_scalar('val_loss', total_val_loss, i)
        writer.add_scalar('running_val_loss', running_val_history[-1], i)
        writer.add_scalar('running_loss', running_history[-1], i)
        writer.add_scalar('train_time', train_time, i)
        writer.add_scalar('val_time',   val_time, i)

        # update the scheduler
        scheduler.step(total_val_loss)

        # print the current condition
        print(f'epoch: {i}: train_loss/ time = {total_loss:.5e}/ {train_time:.2f}s; val_loss/ time = {total_val_loss:.5e}/ {val_time:.2f}s')

        # checkpoint
        total_val_loss = val_loss_history[-1]
        if i % 10 == 1:
            model_name = f"epoch={i}-val_loss={total_val_loss:.5f}.ckpt"
            ckpt_name = os.path.join(checkpoint_version_dir, model_name)

            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'running_history': running_history,
                'running_val_history': running_val_history,
                'model': model,
            },
                ckpt_name
            )

#             torch.save(model, ckpt_name)
            print(f"[checkpointing] = {ckpt_name}")

            figname = os.path.join(checkpoint_version_dir, f"epoch={i}.png")


            if isinstance(dl_val, list):
                for j, dl in enumerate(dl_val):
                    t, X = next(iter(dl))
                    figname = os.path.join(checkpoint_version_dir, f"dl{j}_epoch={i}.png")
                    plot_example(model, t*t_mult, X, figname, writer, i)
                    write_embedding(model, t*t_mult, X, writer, i, j)

            else:
                t, X = next(iter(dl_val))
                plot_example(model, t*t_mult, X, figname, writer, i)
                write_embedding(model, t*t_mult, X, writer, i, j)

        if i <lookback_period: continue

        if early_stopping(running_val_history, lookback_period):
            # print loss curves
            f, axes = plt.subplots(1,2)
            axes.flat[0].loglog(train_loss_history, label='train loss')
            axes.flat[0].loglog(test_loss_history, label='test loss')
            axes.flat[0].legend()
            axes.flat[1].loglog(running_history, label='running train')
            axes.flat[1].loglog(running_val_history, label='running test')
            axes.flat[1].legend()
            plt.tight_layout()

            plt.savefig(os.path.join(checkpoint_version_dir, f"epoch={i}.png"))
            print('===========================================')
            print(f"training stoppped at epoch: {i}")
            break


def early_stopping(running_val_history, lookback_period):
    if running_val_history[-1] > max(running_val_history[-2-lookback_period:-1]):
        return True
    return False

def plot_example(model, t, X, filename,  writer=None, step=None):
    sp_names = ['H2I', 'H2II', 'HI', 'HII', 'HM', 'HeI', 'HeII', 'HeIII', 'de', 'ge']
    f,axes = plt.subplots(2,model.nspecies//2,figsize=(20,10))
    with torch.no_grad():
        abund_paths, loss_path, loss_recon = model(t,X)
    for i in range(model.nspecies):
        for j in range(6):
            axes.flat[i].loglog(t[j], X[j,i,:].detach(), c = f"C{j}")
            if i == 2:
                axes.flat[i].plot(t[j], 0.76 - abund_paths[j,0,:].detach(), marker='.', c = f"C{j%6}", alpha=0.5)
            else:
                axes.flat[i].plot(t[j], abund_paths[j,i,:].detach(), marker='.', c = f"C{j%6}", alpha=0.5)
            axes.flat[i].plot(t[j], abund_paths[j,i,:].detach(), marker='.', c = f"C{j%6}", alpha=0.5)

        axes.flat[i].set_title(sp_names[i])
    plt.tight_layout()
    f.savefig(filename)

    if writer is not None:
        writer.add_figure(filename, f, step)

def write_embedding(model, t, X, writer, step, ith):
    with torch.no_grad():
        # abundance as feature
        feats = model.encoder.log_normalize(X.permute(2,0,1)).reshape(-1,model.nspecies)

        # latent vector from decoder
        writer.add_embedding(feats, tag=f"log_norm_abundances_{ith}", global_step=step)

        x0 = X[:,:,0]
        z0     = model.encode_initial_condition(x0)
        z_path = model.integrate_ODE(t[0]*1e-13, z0).reshape(-1, model.latent_dim)
        writer.add_embedding(z_path, tag=f'latent z space {ith}', global_step=step)

def get_checkpoint_directory(checkpoint_dir='logger_dir'):
    i = 0
    checkpoint_version_dir = os.path.join(checkpoint_dir, f"version_{i}")
    while os.path.exists(checkpoint_version_dir):
        i += 1
        checkpoint_version_dir = os.path.join(checkpoint_dir, f"version_{i}")
    os.makedirs(checkpoint_version_dir)
    return checkpoint_version_dir

if __name__ == "__main__":
    # abundance mean
    log_mean_list = np.array([ -2.7338, -15.3522,  -0.1207,  -7.1836, -13.9073, -17.4677,  -9.0518,
                              -18.2267,  -7.1817,  10.8056])
    log_std_list = np.array([0.2888, 0.5356, 0.0056, 0.7999, 0.6158, 1.8497, 0.1048, 0.2287, 0.7949,
                             0.1916])
    hyperparams = {}

    # dataloader related
    hyperparams['batch_size']       = 32
    hyperparams['train_proportion'] = 0.01
    hyperparams['nworkers'] = 40
    hyperparams['log_mean_list'] = torch.from_numpy(log_mean_list)
    hyperparams['log_std_list']  = torch.from_numpy(log_std_list)
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

    # get appropriate direcotry
    ckpt_dir = get_checkpoint_directory("log_tensorboard")

    dl_train5, dl_test5, dl_val5 = prepare_dataset(dataname='new_dd0053_chemistry_5.hdf5',
                                                   train_proportion=hyperparams['train_proportion'],
                                                   sample_score_file='ds5_score.pt',
                                                   batch_size=hyperparams['batch_size'],
                                                   num_workers=hyperparams['nworkers'])
    dl_train4, dl_test4, dl_val4 = prepare_dataset(dataname='new_dd0053_chemistry_4.hdf5',
                                                   train_proportion=hyperparams['train_proportion'],
                                                   sample_score_file='ds4_score.pt',
                                                   batch_size=hyperparams['batch_size'],
                                                   num_workers=hyperparams['nworkers'])
    dl_train3, dl_test3, dl_val3 = prepare_dataset(dataname='new_dd0053_chemistry_3.hdf5',
                                                   train_proportion=hyperparams['train_proportion'],
                                                   sample_score_file='ds3_score.pt',
                                                   batch_size=hyperparams['batch_size'],
                                                   num_workers=hyperparams['nworkers'])

    dl_train = [dl_train3, dl_train4, dl_train5]
    dl_test  = [dl_test3,  dl_test4,  dl_test5 ]
    dl_val   = [dl_val3,   dl_val4,   dl_val5  ]

    model    = prepare_model(hyperparams['log_mean_list'].numpy(),
                             hyperparams['log_std_list'].numpy(),
                             nspecies = hyperparams['nspecies'],
                             latent_dim= hyperparams['latent_dim'],
                             nhidden=hyperparams['nhidden'],
                             nlayers=hyperparams['nlayers'],
                             activation_fn = hyperparams['activation_fn'],
                             use_plain_decoder=hyperparams['use_plain_decoder'])
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr= hyperparams['initial_lr'] )
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, min_lr=hyperparams['min_lr'])

    del hyperparams['log_mean_list']
    del hyperparams['log_std_list']
    # register hyperparameters here
    writer = SummaryWriter(ckpt_dir)
    writer.add_hparams(hyperparams, {})

    t, X = next(iter(dl_train3))
    writer.add_graph(model.encoder, X[:,:,0])

    z = torch.randn(1,3).double()
    t = torch.zeros(1).double()
    writer.add_graph(model.rhs_func, (t, z))


#    z0 = torch.randn(1,3).double()
#    x  = torch.randn(1,10).double()
#    writer.add_graph(model.decoder, (z, z0, x))

    train_loop(model, [dl_train, dl_test, dl_val], scheduler, writer, run_avg_beta = hyperparams['run_avg_beta'],
               max_epochs=hyperparams['max_epochs'], t_mult = hyperparams['t_mult'], checkpoint_version_dir = ckpt_dir, lookback_period = hyperparams['lookback_period'])


