import torch
from torchdiffeq import odeint

import h5py
import glob
import numpy as np
import time

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, Subset,WeightedRandomSampler, ConcatDataset, SubsetRandomSampler, Sampler
from sklearn.model_selection import train_test_split
from numpy.random import default_rng

from neuralODE.dataloader import H5ODEDataset, H5ODEDatasetParallel
from neuralODE.model import LatentODEfunc, LatentODEfunc_PlainDec


from torch.optim.lr_scheduler import ReduceLROnPlateau

import os

# tensorboard setup
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

RNG_SEED = 42

from numpy.random import default_rng

rng = default_rng(RNG_SEED)


cmap = plt.cm.get_cmap('tab20')
sp_names = ['H2I', 'H2II', 'HI', 'HII', 'HM', 'HeI', 'HeII', 'HeIII', 'de', 'ge']

def get_default_hyperparams():
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
    hyperparams['normalize_abundance'] = True
    return hyperparams

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    ds = worker_info.dataset
    ds.h5py_worker_init()


def prepare_parallel_dataloader(filename='output_file.h5',
                               train_proportion=0.1,
                               batch_size=128,
                               num_workers=40,
                               normalize_abundance=True,
                               normalize_by_tff = False,
                               sample_weight_file=None,
                                ):

    ds = H5ODEDatasetParallel(filename,
                    normalize_abundance=normalize_abundance,
                    normalize_by_tff=normalize_by_tff,)


    nsamples = int( len(ds)* train_proportion )
    subset_indexes = rng.choice(len(ds),nsamples)

    idx_train, idx_val_test = train_test_split(subset_indexes, test_size=0.2, random_state=RNG_SEED)
    idx_val  , idx_test     = train_test_split(idx_val_test, test_size=0.5, random_state=RNG_SEED)

    if sample_weight_file is not None:
        weights = np.load(sample_weight_file)
        assert len(weights) == len(ds), f"weight file {sample_weight_file} should have the same number of datapoints as dataset {h5filename}"
        weights = torch.from_numpy(weights[idx_train])
        train_sampler = WeightedRandomSampler(weights, len(idx_train))
    else:
        train_sampler = SubsetRandomSampler(idx_train)



    dl_train = DataLoader(ds,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        worker_init_fn = worker_init_fn,
                        collate_fn=collate_multi_paths,
                          pin_memory=True,
                        sampler=train_sampler)
    dl_val  = DataLoader(ds,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        worker_init_fn = worker_init_fn,
                        collate_fn=collate_multi_paths,
                        sampler=SequentialSubsetSampler(idx_val))
    dl_test  = DataLoader(ds,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        worker_init_fn = worker_init_fn,
                        collate_fn=collate_multi_paths,
                        sampler=SequentialSubsetSampler(idx_test))
    return dl_train, dl_test, dl_val

def prepare_concat_dataloaders(dataname=['new_dd0053_chemistry_5.hdf5'],
                               train_proportion=0.1,
                               batch_size=128,
                               num_workers=40,
                               sample_score_file=None,
                               normalize_abundance=True,
                               normalize_by_tff = False,
                               ):
    assert 0 < train_proportion  <= 1, 'proportion of training data is bound betwen 0,1'
    RNG_SEED=42
    rng = default_rng(RNG_SEED)

    ds_list = [H5ODEDataset(d,
                            normalize_abundance=normalize_abundance,
                            normalize_by_tff=normalize_by_tff)
               for d in dataname]
    ds = ConcatDataset(ds_list)

    nsamples = int( len(ds)* train_proportion )
    subset_indexes = rng.choice(len(ds),nsamples)

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


    dl_train = DataLoader(ds_train, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,
                          sampler=train_sampler, collate_fn=collate_multi_paths)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          collate_fn=collate_multi_paths)
    dl_val  = DataLoader(ds_val,  batch_size=batch_size,
                         shuffle=False, num_workers=num_workers,
                         collate_fn=collate_multi_paths)

    return dl_train, dl_test, dl_val

def prepare_dataset(dataname='new_dd0053_chemistry_5.hdf5',
                    train_proportion=0.1, batch_size=128, num_workers=40,
                    sample_score_file=None, normalize_abundance=True,
                    sample_weight_file=None):
    assert 0 < train_proportion  <= 1, 'proportion of training data is bound betwen 0,1'
    RNG_SEED=42
    rng = default_rng(RNG_SEED)

    ds = H5ODEDataset(dataname, normalize_abundance=normalize_abundance)

    nsamples = int( len(ds)* train_proportion )
    subset_indexes = rng.choice(len(ds),nsamples)

    idx_train, idx_val_test = train_test_split(subset_indexes, test_size=0.2, random_state=RNG_SEED)
    idx_val  , idx_test     = train_test_split(idx_val_test, test_size=0.5, random_state=RNG_SEED)

    ds_train = Subset(ds, idx_train)
    ds_val   = Subset(ds, idx_val)
    ds_test  = Subset(ds, idx_test)

    # select the weights from idx
    # score is the output log p(x)
    if sample_score_file is not None:
        score = torch.load(sample_score_file)
        assert len(score) == len(ds), f"weight file {sample_score_file} should have the same number of datapoints as dataset {h5filename}"
        train_score   = torch.from_numpy(score[idx_train])
        weights = torch.maximum( torch.exp(-0.01*train_score), torch.tensor([1e-3]) )
        train_sampler = WeightedRandomSampler(weights, len(idx_train))
        shuffle= False
    else:
        train_sampler = None
        shuffle = True

    if sample_weight_file is not None:
        weights = np.load(sample_weight_file)
        assert len(weights) == len(ds), f"weight file {sample_weight_file} should have the same number of datapoints as dataset {h5filename}"
        weights = torch.from_numpy(weights[idx_train])
        train_sampler = WeightedRandomSampler(weights, len(idx_train))
        shuffle= False
    else:
        train_sampler = None
        shuffle = True

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=train_sampler)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_val  = DataLoader(ds_val,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dl_train, dl_test, dl_val


def prepare_dataloaders(filenames, hyperparams, sample_score_file=None,
                        sample_weight_file=None):
    if sample_score_file is None:
        sample_score_file = [None] * len(filenames)
    assert len(sample_score_file) == len(filenames)

    if sample_weight_file is None:
        sample_weight_file = [None] * len(filenames)
    assert len(sample_weight_file) == len(filenames)

    dl_train, dl_test, dl_val = [], [], []
    for fn, score in zip(filenames, sample_score_file, sample_weight_file):
        dltrain, dltest, dlval =  prepare_dataset(dataname=fn,
                                                   train_proportion=hyperparams['train_proportion'],
                                                   normalize_abundance=hyperparams['normalize_abundance'],
                                                   sample_score_file=score,
                                                   sample_weight_file=weight,
                                                   batch_size=hyperparams['batch_size'],
                                                   num_workers=hyperparams['nworkers'])
        dl_train.append(dltrain)
        dl_test.append(dltest)
        dl_val.append(dlval)
    return dl_train, dl_test, dl_val

def plot_example(model, t, X, index=None, nlines=6, axes=None, field="data"):
    assert field in ['data', 'error', 'log_error']

    nrows = 2
    ncols = 10 // nrows

    if axes is None: f,axes = plt.subplots(nrows, ncols,figsize=(20,10))
    with torch.no_grad():
        X_recon = model(X.permute(0,2,1))
        if index is not None: X_recon = X_recon[index]
        for i in range(10):
            for j in range(nlines):
                idx1, idx2 = i // ncols, i %ncols
                if field == 'data':
                    axes[idx1][idx2].loglog(t[j], X[j,i,:].detach(), c = cmap(j))
                    axes[idx1][idx2].plot(t[j], X_recon[j,:,i].detach(), marker='.', c = cmap(j), alpha=0.5, ls ='')
                elif field == 'error':
                    axes[idx1][idx2].loglog(t[j], (X_recon[j,:,i] - X[j,i,:]).abs() / X[j,i,:] , c = cmap(j))
                    axes[idx1][idx2].set_ylim(1e-8,1e1)
                    axes[idx1][idx2].axhline(1e-2, color='k', alpha=0.4, ls ='--')
                elif field == 'log_error':
                    logX_recon = model.enc.log_normalize(X_recon)
                    logX       = model.enc.log_normalize(X.permute(0,2,1))
                    axes[idx1][idx2].loglog(t[j], ((logX_recon - logX) / logX).abs().numpy()[j,:,i] , c = cmap(j))
                axes[idx1][idx2].set_title(sp_names[i])
        plt.tight_layout()
    return axes


def collate_multi_paths(data):
    tiny = 1e-10
    ts, Xs = zip(*data)

    Xall = torch.stack(Xs)
    tall = torch.stack(ts)

    bsz, nspecies, ntime = Xall.shape

    tflat = tall.flatten()
    tindx = torch.argsort(tflat)
    tunique = torch.unique(tflat)

    # where each observations is at on its 'original time axis' [0:74]
    taxis_indx = torch.from_numpy(np.digitize(tflat, tunique-tiny))[tindx]-1
    # time ordered observations index
    obs_indx = torch.tensor([[i]*ntime for i in range(bsz)]).flatten()[tindx]

    # batch ground truths,
    # time ordered with interwinding observations over different batch element
    # the corresponding actual observeed batch index is given by obs_indx
    X_truth = Xall.permute(0,2,1).reshape(-1,10)[tindx]
    t_X     = tflat[tindx]

    X0 = Xall[:,:,0]

    return tunique, X0, X_truth, obs_indx, taxis_indx, t_X


def compute_log_difference(X,X_pred, mean=True):
    if mean:
        return torch.log(X/X_pred).abs().mean((0,2))
    return torch.log(X/X_pred).abs()

def compute_abund_difference(X,X_pred, mean=True):
    density = (X[:,5,0]/0.24).unsqueeze(1).unsqueeze(2)
    X[:,:5,:]   /= density
    X[:,6:-1,:] /= density

    X_pred[:,:5,:]   /= density
    X_pred[:,6:-1,:] /= density
    if mean:
        return ((X -X_pred) / X).abs().mean((0,2))
    return ((X -X_pred) / X).abs()

def compute_log_difference_norm(X,X_pred, mean=True):
    """for normed abundance data"""
    density = (X[:,5,0]/0.24).unsqueeze(1).unsqueeze(2)
    X[:,:5,:]   *= density
    X[:,6:-1,:] *= density

    X_pred[:,:5,:]   *= density
    X_pred[:,6:-1,:] *= density
    if mean:
        return torch.log(X/X_pred).abs().mean((0,2))
    return torch.log(X/X_pred).abs()

def compute_abund_difference_norm(X,X_pred, mean=True):
    """for normed abundance data"""
    if mean:
        return ((X -X_pred) / X).abs().mean((0,2))
    return ((X -X_pred) / X).abs()

def get_accum_error(model, dataloader, hyperparams, mean=True):
    log_diff = []
    abund_pct_diff = []
    if not hyperparams['normalize_abundance']:
        with torch.no_grad():
            for t, *X in dataloader:
                X_pred, _, _ = model.forward_time_batch(t*hyperparams['t_mult'], *X[:-1])
                X = X[2]
                log_diff.append(compute_log_difference(X, X_pred, mean=mean))
                abund_pct_diff.append(compute_abund_difference(X, X_pred, mean=mean))
    else:
        with torch.no_grad():
            for t, *X in dataloader:
                X_pred, _, _ = model.forward_time_batch(t*hyperparams['t_mult'], *X[:-1])
                X = X[2]
                log_diff.append(compute_log_difference_norm(X, X_pred, mean=mean))
                abund_pct_diff.append(compute_abund_difference_norm(X, X_pred, mean=mean))
    if mean:
        return torch.stack(log_diff).mean(0), torch.stack(abund_pct_diff).mean(0)
    return torch.stack(log_diff), torch.stack(abund_pct_diff)

def get_error_dataloaders(m1, dataloader_list):
    e1, e2 = [], []
    total_d = 0
    for d in dataloader_list:
        log_diff, abund_pct_diff = get_accum_error(m1,d)
        e1.append(log_diff* len(d))
        e2.append(abund_pct_diff* len(d))
        total_d += len(d)
    return sum(e1)/total_d, sum(e2)/total_d


class SequentialSubsetSampler(Sampler):
    def __init__(self, indices) -> None:
        self.indices = indices
        self.idx = 1
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)
