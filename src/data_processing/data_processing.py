import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

class NumpyDataset(Dataset):
    def __init__(self, numpy_array):
        self.data = numpy_array.astype(np.float32)  # convert once, but stay in numpy

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])  # convert per sample on-the-fly

def get_data_loader(train_data, batch_size, shuffle=True):
    dataset = NumpyDataset(train_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)