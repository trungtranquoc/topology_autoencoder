import numpy as np
import torch   
from torch.utils.data import DataLoader, Dataset

class ClassificationDataset(Dataset):
    def __init__(self, X, y = None):
        self.X = X.astype(np.float32)  # convert once, but stay in numpy
        if y is not None:
            self.y = y.astype(np.int64)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])  # convert per sample on-the-fly
        else:
            return torch.from_numpy(self.X[idx])
        
    def preprare_batch(self, batch_size=64):
        return DataLoader(self, batch_size=batch_size)