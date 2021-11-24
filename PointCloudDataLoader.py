import os
import h5py
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from glob import glob
from torch.utils.data import Dataset


def load_data(dir, partition="train"):
    all_data = []
    all_label = []
    datapath = 'ply_data_%s*.h5'%partition
    for h5_name in glob(os.path.join(dir, datapath)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        if 'normal' in f:
            normal = f['normal'][:].astype('float32')
        else:
            normal = np.zeros_like(data)
        f.close()
        data = np.concatenate([data,normal], axis=-1)
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class ModelNetDataLoader(Dataset):
    def __init__(self, opts, partition='train'):
        self.opts = opts
        self.data, self.label = load_data(opts.data_dir, partition)
        self.num_points = opts.num_points
        self.partition = partition
        self.dim = 6 if self.opts.use_normal else 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pc = self.data[index][:self.num_points,:self.dim].copy()
        label = self.label[index]
        return pc.astype(np.float32), label.astype(np.int32)
