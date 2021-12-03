import torch
from torchdiffeq import odeint
import sys

sys.path.append('../')

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
from neuralODE._model import NewLatentODEfunc
from neuralODE.utils import *


from torch.optim.lr_scheduler import ReduceLROnPlateau

import os

# tensorboard setup
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

import argparse

import wandb


RNG_SEED = 42

np.random.seed(RNG_SEED)


def train_step(model, optimizer, dataloaders, t_mult, max_time = None,
               get_pct_diff = None):
    global device
    total_loss = 0.0
    time0 = time.time()
    for t, X0, X_truth, obs_indx, taxis_indx, t_X in dataloaders:
        t, X0, X_truth, obs_indx, taxis_indx, t_X = t.to(device, non_blocking=True), X0.to(device, non_blocking=True), X_truth.to(device, non_blocking=True), obs_indx.to(device, non_blocking=True), taxis_indx.to(device, non_blocking=True), t_X.to(device, non_blocking=True)

        flag_t  = torch.ones_like(t, dtype=torch.bool).to(device)
        flag_tX = torch.ones_like(t_X, dtype=torch.bool).to(device)
        if max_time is not None:
            flag_t  = t <= max_time
            flat_tX = t_X <= max_time
        optimizer.zero_grad()
        abund_paths, loss_path, loss_recon = model.forward_time_batch(t[flag_t]* t_mult,X0,
                                                                      X_truth[flat_tX],
                                                                      obs_indx[flat_tX],
                                                                      taxis_indx[flat_tX],
                                                                      get_pct_diff=get_pct_diff)
        loss = loss_path + loss_recon
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item() / len(dataloaders)
        if torch.isnan(loss):
            print('failedd loss', loss, loss_path, loss_recon)
            raise ValueError
    train_time = time.time() - time0
    return total_loss, train_time

def val_step(model, dataloaders, t_mult, max_time = None, get_pct_diff=None):
    global device
    with torch.no_grad():
        total_loss = 0.0
        time0 = time.time()
        for t, X0, X_truth, obs_indx, taxis_indx, t_X in dataloaders:
            t, X0, X_truth, obs_indx, taxis_indx, t_X = t.to(device), X0.to(device), X_truth.to(device), obs_indx.to(device), taxis_indx.to(device), t_X.to(device)
            flag_t  = torch.ones_like(t, dtype=torch.bool)
            flag_tX = torch.ones_like(t_X, dtype=torch.bool)
            if max_time is not None:
                flag_t  = t <= max_time
                flat_tX = t_X <= max_time
            abund_paths, loss_path, loss_recon = model.forward_time_batch(t[flag_t]* t_mult,X0,
                                                                          X_truth[flat_tX],
                                                                          obs_indx[flat_tX],
                                                                          taxis_indx[flat_tX],
                                                                          get_pct_diff=get_pct_diff)
            loss = loss_path + loss_recon
            total_loss += loss.item() / len(dataloaders)
        val_time = time.time() - time0
    return total_loss, val_time


def train_loop(model, dataloaders, scheduler, writer, run_avg_beta = 0.99,
               max_epochs=10000, t_mult = 1e-11, checkpoint_version_dir = '.',
               lookback_period = 20, max_time=None,
               wandb_logger=None,get_pct_diff=None):


    train_loss_history, val_loss_history     = [], []
    running_history,    running_val_history  = [], []

    β = run_avg_beta

    dl_train, dl_test, dl_val = dataloaders

    time_between_ckpts = 0.0

    for i in range(max_epochs):
        # training and validation step

#        with torch.profiler.profile(
#                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                    #on_trace_ready=torch.profiler.tensorboard_trace_handler(checkpoint_version_dir),
                    #record_shapes=True,
                    #profile_memory=True,
                    #with_stack=True,
        #) as prof:
        total_loss, train_time = train_step(model, optimizer, dl_train, t_mult,
                                            max_time,get_pct_diff)

         #   prof.step()


        total_val_loss, val_time = val_step(model, dl_test, t_mult, max_time,
                                            get_pct_diff)


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

        if wandb_logger is not None:
            metrics = {}
            metrics['training_loss'] = total_loss
            metrics['val_loss']      = total_val_loss
            metrics['running_val_loss'] = running_val_history[-1]
            metrics['running_loss']     = running_history[-1]
            metrics['train_time']       = train_time
            metrics['val_time']         = val_time
            wandb.log(metrics)


        # update the scheduler
        scheduler.step(total_val_loss)

        # print the current condition
        print(f'epoch: {i}: train_loss/ time = {total_loss:.5e}/ {train_time:.2f}s; val_loss/ time = {total_val_loss:.5e}/ {val_time:.2f}s')

        # if accumulate train time
        time_between_ckpts += train_time + val_time

        # prompt checkpoint if there arent checpoints for an hour
        total_val_loss = val_loss_history[-1]
        if i % 10 == 1 or time_between_ckpts > 3600:
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
                'hyperparams': hyperparams,
            },
                ckpt_name
            )
            if wandb_logger is not None:
                wandb.save(ckpt_name)

#             torch.save(model, ckpt_name)
            print(f"[checkpointing] = {ckpt_name}")

            figname = os.path.join(checkpoint_version_dir, f"epoch={i}.png")
            plot_example(model, dl_val, figname, t_mult, writer, i,
                         norm_by_density= not hyperparams['normalize_abundance'],
                         max_time=max_time,
                         wandb_logger=wandb_logger)

            figname = os.path.join(checkpoint_version_dir, f"_err_epoch={i}.png")
            plot_example(model, dl_val, figname, t_mult, writer, i,
                         norm_by_density= not hyperparams['normalize_abundance'], max_time=max_time,
                         wandb_logger=wandb_logger, plot_type='error')

            figname = os.path.join(checkpoint_version_dir, f"_err_dist_epoch={i}.png")
            plot_error_distribution(model, dl_val, figname, t_mult, writer, i,
                                    norm_by_density= not hyperparams['normalize_abundance'], max_time=max_time,
                                    wandb_logger=wandb_logger)

            #write_embedding(model, t*t_mult, X, writer, i, j)

            # reset checkpoint times
            time_between_ckpts = 0.0

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

def plot_example(model, dataloader, filename, t_mult=1e-13,  writer=None, step=None,
                 norm_by_density=False, max_time=None, wandb_logger=None,
                 plot_type = 'data'):
    global device
    t, X0, X_truth, obs_indx, taxis_indx, t_X = next(iter(dataloader))
    t, X0, X_truth, obs_indx, taxis_indx, t_X = t.to(device), X0.to(device), X_truth.to(device), obs_indx.to(device), taxis_indx.to(device), t_X.to(device)
    sp_names = ['H2I', 'H2II', 'HI', 'HII', 'HM', 'HeI', 'HeII', 'HeIII', 'de', 'ge']

    sharey =True if plot_type == 'error' else False
    f,axes = plt.subplots(2,model.nspecies//2,figsize=(20,10), sharey=sharey)

    bsz, _ = X0.shape
    with torch.no_grad():
        flag_t  = torch.ones_like(t, dtype=torch.bool)
        flag_tX = torch.ones_like(t_X, dtype=torch.bool)
        if max_time is not None:
            flag_t  = t <= max_time
            flat_tX = t_X <= max_time
        optimizer.zero_grad()
        abund_paths, loss_path, loss_recon = model.forward_time_batch(t[flag_t]* t_mult,X0,
                                                                      X_truth[flat_tX],
                                                                      obs_indx[flat_tX],
                                                                      taxis_indx[flat_tX])

    for i in range(model.nspecies):
        for j in range(10):
            flag = (torch.where(obs_indx[flat_tX] == j)[0]).cpu()
            density = X_truth[flag[0],5]/0.24 if norm_by_density and i not in [5, 9] else torch.ones(1)
            taxis = t_X[flag].cpu()*t_mult
            X_true = X_truth[flag,i].cpu().detach()/ density.cpu()
            X_pred = abund_paths[flag,i].cpu().detach()/density.cpu()

            if plot_type=='data':
                axes.flat[i].loglog(taxis, X_true , c = f"C{j}")
                axes.flat[i].plot(taxis, X_pred, marker='.', c = f"C{j%6}", alpha=0.5)
            else:
                pct_err = (X_true - X_pred).abs() / X_true
                axes.flat[i].loglog(taxis, pct_err, c = f"C{j}")
                axes.flat[i].plot([0, max(taxis)], [1e-2,1e-2],color='k', ls='--')

        axes.flat[i].set_title(sp_names[i])
    plt.tight_layout()
    f.savefig(filename)

    if writer is not None:
        writer.add_figure(filename, f, step)

    if wandb_logger is not None:
        wandb.log({"plots"+plot_type: f})
    plt.close(f)

def write_embedding(model, t, X, writer, step, ith):
    with torch.no_grad():
        # abundance as feature
        feats = model.autoencoder.enc.log_normalize(X.permute(2,0,1)).reshape(-1,model.nspecies)

        # latent vector from decoder
        writer.add_embedding(feats, tag=f"log_norm_abundances_{ith}", global_step=step)

        x0 = X[:,:,0]
        z0     = model.encode_initial_condition(x0)
        z_path = model.integrate_ODE(t[0]*1e-13, z0).reshape(-1, model.latent_dim)
        writer.add_embedding(z_path, tag=f'latent z space {ith}', global_step=step)

def plot_error_distribution(model,
                            dataloader,
                            filename,
                            t_mult=1e-13,
                            writer=None,
                            step=None,
                            norm_by_density=False,
                            max_time=None,
                            wandb_logger=None):
    all_pct_diff       = [np.zeros(99) for i in range(10)]
    all_abund_pct_diff = [np.zeros(99) for i in range(10)]
    bins = np.logspace(-6,1,100)

    for t, X0, X_truth, obs_indx, taxis_indx, t_X in dataloader:
        global device
        t, X0, X_truth, obs_indx, taxis_indx, t_X = t.to(device), X0.to(device), X_truth.to(device), obs_indx.to(device), taxis_indx.to(device), t_X.to(device)
        X = [X0, X_truth, obs_indx, taxis_indx, t_X]

        flag_t  = torch.ones_like(t, dtype=torch.bool).to(device)
        flag_tX = torch.ones_like(t_X, dtype=torch.bool).to(device)

        if max_time is not None:
            flag_t  = t <= max_time
            flat_tX = t_X <= max_time
        with torch.no_grad():
            X_pred, loss_path, loss_recon = model.forward_time_batch(t[flag_t],X0,
                                                                   X_truth[flat_tX],
                                                                   obs_indx[flat_tX],
                                                                   taxis_indx[flat_tX])
        # prepare data to within max_time
        X_T = X_truth[flat_tX]
        pct_diff = (X_T - X_pred) / X_T

        density = X0[:,5]
        X_pred[:,:-1] = X_pred[:,:-1] / (density[obs_indx[flat_tX]].unsqueeze(1))

        # prepare data to within max_time
        X_T = X_truth[flat_tX]
        X_T[:,:-1] = X_T[:,:-1] / (density[obs_indx[flat_tX]].unsqueeze(1))

        abund_pct_diff = (X_T - X_pred) / X_T

        for i in range(model.nspecies):
            pct_counts,   binc = np.histogram(pct_diff.cpu().abs().numpy().T[i], bins)
            abund_counts, binc = np.histogram(abund_pct_diff.cpu().abs().numpy().T[i], bins)
            all_pct_diff[i]       += pct_counts
            all_abund_pct_diff[i] += abund_counts

    f,axes = plt.subplots(2,model.nspecies//2,figsize=(20,10),sharex=True, sharey=True)
    sp_names = ['H2I', 'H2II', 'HI', 'HII', 'HM', 'HeI', 'HeII', 'HeIII', 'de', 'ge']
    ax = axes.flat
    binc = 0.5*(bins[:-1] + bins[1:])
    for i in range(10):
        ax[i].plot(binc, all_pct_diff[i], alpha=0.3, color=f'C{i}' )
        ax[i].plot(binc, all_abund_pct_diff[i], alpha=0.8, color=f'C{i}' )

        xloc =  binc[np.argmax(all_pct_diff[i])]
        ax[i].plot( [xloc, xloc], [0, max(all_abund_pct_diff[i])+10 ], color=f'C{i}', ls='--')
        ax[i].plot( [1e-2, 1e-2], [0, max(all_abund_pct_diff[i])+10 ], color='k', ls='--')
        #ax[i].axvline( binc[np.argmax(all_abund_pct_diff[i])], color=f'C{i}', ls='--')

        #ax[i].axvline(1e-2, color='k', ls='--')
        ax[i].set_xscale('log')
        ax[i].set_title(sp_names[i])

    plt.tight_layout()
    f.savefig(filename)

    if writer is not None:
        writer.add_figure(filename, f, step)

    if wandb_logger is not None:
        wandb.log({"plots_error_dist": f})

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


    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument( '--norm_tff' , '-normalize_by_tff',
                        help='normalize the time axis by free fall time',
                        default=False,
                        action='store_true'
                        )

    parser.add_argument('--nhidden' ,'-nhidden',
                        type=int,
                        default=32,
                        help='number of hidden unit used in the MLP')
    parser.add_argument('--t_mult' ,'-t_mult',
                        type=float,
                        default = 1e-13,
                        help='constant factor to multiply the time axis with')
    parser.add_argument('--norm_abund' ,'-normalize_by_abundance',
                        default = False,
                        help='normalize the abundance by density',
                        action='store_true')
    parser.add_argument('--ae_type' ,'-autoencoder',
                        type=str,
                        default = 'ICAE',
                        help='type of autoencoder',
                        choices= ['PlainAE','ICAE','ICHAE']
                        )
    parser.add_argument('--train_proportion',
                        type=float,
                        default = 1.0,
                        help='proportion of data to use',
                        )
    parser.add_argument('--batch_size',
                        type=int,
                        default = 32,
                        help='batch size of the dataloader',
                        )
    parser.add_argument('--max_time',
                        type=float,
                        default = 1.0,
                        help='maximum evolve time for training',
                        )
    parser.add_argument('--max_epoch',
                        type=int,
                        default = 10000,
                        help='maximum training epoch to run',
                        )
    parser.add_argument('--pct_diff',
                        default = False,
                        action='store_true',
                        help='calculate the percentage difference as loss',
                        )
    parser.add_argument('--nlayers',
                        type=int,
                        default = 4,
                        help='number of layers in all MLPs',
                        )
    parser.add_argument('--nlayers_ode',
                        type=int,
                        default = 4,
                        help='number of layers in the MLP for the neural ODE',
                        )
    args = parser.parse_args()

    if args.norm_abund:
        log_mean_list = np.array([ -2.7338, -15.3522,  -0.1207,  -7.1836, -13.9073, -17.4677,  -9.0518,
                                -18.2267,  -7.1817,  10.8056])
        log_std_list = np.array([0.2888, 0.5356, 0.0056, 0.7999, 0.6158, 1.8497, 0.1048, 0.2287, 0.7949,
                                0.1916])
        # normalization method for HI
        mean = torch.zeros(10).numpy()
        mean[2] = 0.76
        std = torch.ones(10).numpy()
        std[2] = 0.0002
        mask  = torch.zeros(10).numpy()
        mask[2] = 1.0
    else:
        # for NON-normalize mean/std
        log_mean_list = np.array([-21.1365, -33.2490, -18.4069, -24.8810, -31.7336,
                                -18.9038, -27.3975, -36.6503, -24.8808,
                                10.6995], dtype=np.float64)
        log_std_list = np.array([2.5124, 1.6098, 2.1520, 1.2604, 1.4789, 2.1591,
                                2.2599, 2.3831, 1.2658, 0.2462], dtype=np.float64)

        mean = torch.zeros(10).numpy()
        mean[2] = 0.76
        std = torch.ones(10).numpy()
        std[2] = 0.0002
        mask  = torch.zeros(10).numpy()
        mask[2] = 1.0
        #mean = None
        #std  = None
        #mask = None

    hyperparams = get_default_hyperparams()
    hyperparams['log_mean_list'] = log_mean_list
    hyperparams['log_std_list']  = log_std_list
    hyperparams['data_mean']  = mean
    hyperparams['data_std']   = std
    hyperparams['data_mask']  = mask
    hyperparams['nhidden']  = args.nhidden
    hyperparams['train_proportion'] = args.train_proportion
    hyperparams['activation_fn']    = 'ELU'
    hyperparams['nworkers']         = args.batch_size
    hyperparams['normalize_abundance'] = args.norm_abund

    # normalize time axis
    hyperparams['normalize_by_tff'] = args.norm_tff
    hyperparams['t_mult'] = args.t_mult
    hyperparams['max_time'] = args.max_time
    hyperparams['ae_type'] = args.ae_type
    hyperparams['max_epoch'] = args.max_epoch
    hyperparams['pct_diff'] = args.pct_diff
    hyperparams['nlayers'] = args.nlayers
    hyperparams['nlayers_ode'] = args.nlayers_ode

    print(args)

    run = wandb.init(
        project="chemical_kinetics_neuralODE",
        entity="hisunnytang",
        config=hyperparams
    )

    # initialize dataloaders
    filenames = ['data/sample_new_dd0053_chemistry_3.hdf5',
                 'data/sample_new_dd0053_chemistry_4.hdf5',
                 'data/sample_new_dd0053_chemistry_5.hdf5']
    #dl_train, dl_test, dl_val = prepare_dataloaders(filenames, hyperparams, sample_score_file=None)
    dl_train, dl_test, dl_val = prepare_parallel_dataloader("output_file.h5",
                                                           train_proportion=hyperparams['train_proportion'],
                                                           normalize_abundance=hyperparams['normalize_abundance'],
                                                           batch_size=hyperparams['batch_size'],
                                                           num_workers=hyperparams['nworkers'],
                                                           normalize_by_tff=hyperparams['normalize_by_tff'],
                                                           )

# log an artifact
    artifact = wandb.Artifact('combined_dataset', type='dataset')
    artifact.add_file('output_file.h5')
    run.log_artifact(artifact)


    #dl_train, dl_test, dl_val = prepare_concat_dataloaders(filenames,
    #                                                       train_proportion=hyperparams['train_proportion'],
    #                                                       normalize_abundance=hyperparams['normalize_abundance'],
    #                                                       batch_size=hyperparams['batch_size'],
    #                                                       num_workers=hyperparams['nworkers'],
    #                                                       normalize_by_tff=hyperparams['normalize_by_tff'],
    #                                                       )

    # get appropriate direcotry
    ckpt_dir = get_checkpoint_directory("log_wandb")

    print(hyperparams)
    print(ckpt_dir)

    global device
    device = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # initialize model
    model = NewLatentODEfunc(
                 hyperparams['log_mean_list'],
                 hyperparams['log_std_list'],
                 latent_dim=hyperparams['latent_dim'],
                 nlayers=hyperparams['nlayers'],
                 nlayers_ode=hyperparams['nlayers_ode'],
                 nhidden=hyperparams['nhidden'],
                 mixing=1.0,
                 nspecies=hyperparams['nspecies'],
                 use_binaryODE=False,
                 activation_fn = hyperparams['activation_fn'],
                 data_mean=hyperparams['data_mean'],
                 data_std=hyperparams['data_std'],
                 data_mask=hyperparams['data_mask'],
                 ode_final_activation=None,
                 ae_type = hyperparams['ae_type'],
                 map_H2 = False,
                 #ae_type = 'ICHAE'
                 #ae_type = 'PlainAE'

    ).double().to(device)

    print(model)

    wandb.watch(model, log_freq=100)

    optimizer = torch.optim.Adam(model.parameters(), lr=
                                 hyperparams['initial_lr'] )
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  min_lr=hyperparams['min_lr'])


    # register hyperparameters here
    writer = SummaryWriter(ckpt_dir)

    train_loop(model, [dl_train, dl_test, dl_val], scheduler, writer, run_avg_beta = hyperparams['run_avg_beta'],
               max_epochs=hyperparams['max_epochs'], t_mult =
               hyperparams['t_mult'], checkpoint_version_dir = ckpt_dir,
               lookback_period = hyperparams['lookback_period'],
               max_time=hyperparams['max_time'], wandb_logger=True,
               get_pct_diff=hyperparams['pct_diff'])


