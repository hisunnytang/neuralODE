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


def train_step(model, optimizer, dataloaders,):
    global device
    total_loss = 0.0
    time0 = time.time()
    for t, X, mult in dataloaders:
        t, X, mult = t.to(device,non_blocking=True), X.to(device,non_blocking=True), mult.to(device,non_blocking=True)

        optimizer.zero_grad()
        abund_paths, loss_path, loss_recon = model.forward_tff_batch(t, X,
                                                                     mult,
                                                                     weighted_loss=True)

        loss = loss_path + loss_recon
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item() / len(dataloaders)
        if torch.isnan(loss):
            print('failedd loss', loss, loss_path, loss_recon)
            raise ValueError
    train_time = time.time() - time0
    return total_loss, train_time

def val_step(model, dataloaders):
    global device
    with torch.no_grad():
        total_loss = 0.0
        time0 = time.time()
        for t, X, mult in dataloaders:
            t, X, mult = t.to(device,non_blocking=True), X.to(device,non_blocking=True), mult.to(device,non_blocking=True)
            #abund_paths, loss_path, loss_recon = model.forward_tff_batch(t, X, mult)
            abund_paths, loss_path, loss_recon = model.forward_tff_batch(t, X,
                                                                        mult,
                                                                        weighted_loss=True)
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
        total_loss, train_time = train_step(model, optimizer, dl_train,)

         #   prof.step()


        total_val_loss, val_time = val_step(model, dl_test,)


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
    t, X, mult = next(iter(dataloader))
    t, X, mult = t.to(device), X.to(device), mult.to(device)
    sp_names = ['H2I', 'H2II', 'HI', 'HII', 'HM', 'HeI', 'HeII', 'HeIII', 'de', 'ge']

    sharey =True if plot_type == 'error' else False
    f,axes = plt.subplots(2,model.nspecies//2,figsize=(20,10), sharey=sharey)

    with torch.no_grad():
        #abund_paths, loss_path, loss_recon = model.forward_tff_batch(t, X, mult)
        abund_paths, loss_path, loss_recon = model.forward_tff_batch(t, X,
                                                                     mult,
                                                                     weighted_loss=True)

    for i in range(model.nspecies):
        for j in range(10):
            density = X[j,5,0]/0.24 if norm_by_density and i not in [5, 9] else torch.ones(1)
            taxis = t[j] / mult[j]
            X_true = X[j,i,:] / density
            X_pred = abund_paths[j,i,:]/ density

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

    for t, X, mult in dataloader:
        with torch.no_grad():
            #abund_paths, loss_path, loss_recon = model.forward_tff_batch(t,X,mult)
            abund_paths, loss_path, loss_recon = model.forward_tff_batch(t, X,
                                                                        mult,
                                                                        weighted_loss=True)

        X      = X.permute(0,2,1).reshape(-1, model.nspecies)
        X_pred = abund_paths.permute(0,2,1).reshape(-1, model.nspecies)

        pct_diff = ((X- X_pred)/ X).abs().cpu().numpy()

        for i in range(model.nspecies):
            pct_counts,   binc = np.histogram(pct_diff.T[i], bins)
            all_pct_diff[i] += pct_counts

    f,axes = plt.subplots(2,model.nspecies//2,figsize=(20,10),sharex=True, sharey=True)
    sp_names = ['H2I', 'H2II', 'HI', 'HII', 'HM', 'HeI', 'HeII', 'HeIII', 'de', 'ge']
    ax = axes.flat
    binc = 0.5*(bins[:-1] + bins[1:])
    for i in range(model.nspecies):
        ax[i].plot(binc, all_pct_diff[i], alpha=0.3, color=f'C{i}' )

        xloc =  binc[np.argmax(all_pct_diff[i])]
        ax[i].plot( [xloc, xloc], [0, max(all_pct_diff[i])+10 ], color=f'C{i}', ls='--')
        ax[i].plot( [1e-2, 1e-2], [0, max(all_pct_diff[i])+10 ], color='k', ls='--')
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
    parser.add_argument('--latent_dim',
                        type=int,
                        default=3,
                        help='latent dimension of the Neural ODE')

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

    parser.add_argument('--map_HI',
                        default=False,
                        action='store_true',
                        help='map HI differently from the rest',
                        )

    parser.add_argument('--batch_ODE',
                        default=False,
                        action='store_true',
                        help='scale the ODE such that the time axis aligns',
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

        #mean = torch.zeros(10).numpy()
        #mean[2] = 0.76
        #std = torch.ones(10).numpy()
        #std[2] = 0.0002
        #mask  = torch.zeros(10).numpy()
        #mask[2] = 1.0
        mean = None
        std  = None
        mask = None


    # using the data from the full range
    log_mean_list = np.array([-26.89642015, -36.08992384, -22.84191932,
                              -27.55947267, -34.55619354, -23.33877351, -32.45980892,
                              -40.86778628, -27.56028564,  10.5472939 ])
    log_std_list = np.array([4.74021849, 2.73507229, 3.60797819, 2.22993686,
                              2.57588971, 3.61725493, 4.08334541, 3.60757713,
                              2.2358302 , 0.59669775])


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
    hyperparams['map_HI'] = args.map_HI
    hyperparams['latent_dim'] = args.latent_dim

    print(args)

    run = wandb.init(
        project="chemical_kinetics_neuralODE",
        entity="hisunnytang",
        config=hyperparams
    )


    filename = "../VDS.h5"
    weight_file = "../Analysis/VDS_weights.npy" #None # "vds_weights.npy"
    dl_train, dl_test, dl_val = prepare_parallel_dataloader(filename,
                                                           train_proportion=hyperparams['train_proportion'],
                                                           normalize_abundance=hyperparams['normalize_abundance'],
                                                           batch_size=hyperparams['batch_size'],
                                                           num_workers=hyperparams['nworkers'],
                                                           normalize_by_tff=hyperparams['normalize_by_tff'],
                                                           sample_weight_file=weight_file,
                                                            collate_fn= collate_normed_ODE)


    # log an artifact
    artifact = wandb.Artifact('combined_dataset', type='dataset')
    artifact.add_file(filename)
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
                 map_HI = hyperparams['map_HI'],
                 #ae_type = 'ICHAE'
                 #ae_type = 'PlainAE'

    ).double().to(device)

    print(model)

    wandb.watch(model, log_freq=100)

    optimizer = torch.optim.AdamW(model.parameters(), lr=
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


