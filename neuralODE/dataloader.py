from torch.utils.data import Dataset, DataLoader, Subset
import torch
import h5py
import numpy as np
from random import Random
import torch.distributed as dist
from torch.multiprocessing import Process
import os

# https://tuni-itc.github.io/wiki/Technical-Notes/Distributed_dataparallel_pytorch/
# https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
class H5ODEDataset(Dataset):
    def __init__(self, filename, len_time=100, normalize_abundance=False):
        self.filename = filename
        self.len_time = len_time
        #self.sp_names = self.get_species_name()
        self.sp_names = ['H2I', 'H2II', 'HI', 'HII', 'HM', 'HeI', 'HeII', 'HeIII', 'de', 'ge']
        self.length   = self.get_length()
        self.nspecies = len(self.sp_names)
        self.taxis, self.dtf, self.endindex = self.get_timeaxis()
        self.mh = 1.67e-24
        self.normalize_abundance = normalize_abundance
        self.dataset= None

    def get_species_name(self):
        with h5py.File(self.filename, 'r') as f:
            sp_paths = f.attrs['data_columns']
        return list(sorted(sp_paths))

    def get_length(self):
        with h5py.File(self.filename, 'r') as f:
            data_length = f['data_group'].shape[0]
        return int(data_length)

    def get_timeaxis(self):
        with h5py.File(self.filename, 'r') as f:
            dt  = f.attrs['dtf'] / 1e4
            dtf =  f.attrs['dtf']
            taxis = (np.cumsum([0] + list(1.1**np.arange(100))))*dt

        idx = np.where(taxis<=dtf)[0][-1]+1
        taxis[idx] = dtf
        taxis[idx+1:] = 0.0
        return taxis, dtf,  idx


    def normalize(self, data, ith_sp):
        # min-max normalization
        return (np.log10(data) - self.min_list[ith_sp]) / (self.max_list[ith_sp] - self.min_list[ith_sp])

    def plot_normalize_data(self, idx):
        tdata, Xdata = self[idx]
        f,ax = plt.subplots()
        for i, s in enumerate(self.sp_names):
            ax.loglog(tdata, Xdata[i]/ Xdata[i,0], label=s)
        plt.legend()
        return f

    def __len__(self):
        return self.length

    def get_group_stats(self):
        with h5py.File(self.filename, 'r') as f:
            X = f['data_group'][:,:,:self.endindex+1] # n_init, nspecies, ntimes
            if self.normalize_abundance:
                density = X[:,5:6,0:1]/0.24
                X[:,:5,:] /= density
                X[:,6:-1,:] /= density
            logX = np.log10(X)
        return logX.mean(axis=(0,2)), logX.std(axis=(0,2))

    def __getitem__(self, idx):
#         X = np.zeros((self.nspecies, self.len_time))
        if self.dataset is None:
            self.dataset = h5py.File(self.filename, 'r')['data_group']
        X = self.dataset[idx][:,:self.endindex+1]
        if self.normalize_abundance:
            density = X[5:6,0]/0.24
            X[:5] /= density
            X[6:-1] /= density
        return torch.from_numpy(self.taxis[:self.endindex+1]), torch.from_numpy(X)
#         with h5py.File(self.filename, 'r') as f:
#             X = f['data_group'][idx][:,:self.endindex+1]
#             if self.normalize_abundance:
#                 density = X[5:6,0]/0.24
#                 X[:5] /= density
#                 X[6:-1] /= density
#         return torch.from_numpy(self.taxis[:self.endindex+1]), torch.from_numpy(X)


def initialize_dataloaders(h5filename, nsamples, seed=42, test_size=0.2, batch_size=128, num_workers=40, sample_weight_file=None):
    np.random.seed(seed)
    ds = H5ODEDataset(h5filename)

    subset_indexes = np.random.choice(len(ds),nsamples)
    idx_train, idx_test = train_test_split(subset_indexes, test_size=test_size, random_state=seed)

    ds_train = Subset(ds, idx_train)
    ds_test  = Subset(ds, idx_test)



    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,sampler=sampler)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers,sampler=sampler)
    return iter(dl_train), iter(dl_test), ds, ds_train, ds_test

""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset(batch_size=40):
    """ Partitioning MNIST """
    dataset = H5ODEDataset('../new_dd0053_chemistry_5.hdf5')

    np.random.seed(42)
    nsamples = len(dataset)//500
    subset_indexes = np.random.choice(len(dataset),nsamples)

    idx_train, idx_test = train_test_split(subset_indexes, test_size=0.2, random_state=42)
    # idx_train, idx_test = train_test_split(np.arange(1000), test_size=0.2, random_state=42)

    ds_train = Subset(dataset, idx_train)
    ds_test  = Subset(dataset, idx_test)

    size = dist.get_world_size()
    bsz = batch_size // size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(ds_train, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    return train_set, bsz


""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

from .model import LatentODEfunc
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import time

def run(rank, size):
    torch.manual_seed(1234)

    log_mean_list = np.array([-16.96950552, -30.84290244, -14.84300703, -22.84788406,
       -29.44929018, -15.3370473 , -23.66581591, -32.75255965,
       -22.83376837,  10.89580558])
    log_std_list  = np.array([0.5911518 , 0.39522451, 0.28198317, 0.24839182, 0.3290554 ,
       0.29472535, 0.29449144, 0.30011963, 0.24847854, 0.11540283])

    latent_ode_function = LatentODEfunc(log_mean_list,
                                    log_std_list,
                                    latent_dim=4,
                                    nspecies=10, use_binaryODE=False)
    optim = torch.optim.SGD(latent_ode_function.parameters(), lr=1e-3)


    train_set, bsz = partition_dataset()

    t_mult = 1/1e11

    for epoch in range(10):
        epoch_loss = 0.0
        time0 = time.time()
        for t, X in train_set:
            optim.zero_grad()
            # getting density estimate from helium
            #t_mult = (X[0,5:6,0]*6.67e-8 /0.24)**0.5
            abund_paths, loss_path, loss_recon = latent_ode_function(t* t_mult,X)
            loss = loss_path + loss_recon
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(latent_ode_function)
            optim.step()

        if dist.get_rank() == 0:
            print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / 40, 'time taken:', f"{time.time() - time0:.3e}s")

def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 40
    processes = []


    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
