import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, path, mode='train', step=5, scale=10.0):
        self.path = path
        self.scale = scale
        with h5py.File(path, 'r') as f:
            total = f['X'].shape[0]
            split = int(total * 0.8)
            indices = np.arange(0, split) if mode == 'train' else np.arange(split, total)

            # Subsample to target frequency
            start_idx = step - 1 if step > 1 else 0
            self.X = f['X'][indices][:, start_idx::step, :]
            self.Y = f['Y'][indices][:, start_idx::step, :]
            self.ego = f['ego_states'][indices]
            self.neighbors = f['S_delta_p'][indices]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        hist = torch.tensor(self.X[idx] / self.scale, dtype=torch.float32)
        fut = torch.tensor(self.Y[idx] / self.scale, dtype=torch.float32)
        ego = torch.tensor(self.ego[idx], dtype=torch.float32)
        neighbors = torch.tensor(self.neighbors[idx], dtype=torch.float32)
        return hist, fut, ego, neighbors
