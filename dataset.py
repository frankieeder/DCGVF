import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd

class SampleDataset(Dataset):
    def __init__(self, root, ext='.npz'):
        self.root = root
        self.files = []
        for cd, folders, files in os.walk(root):
            self.files.extend([
                os.path.join(cd, f)
                for f
                in files
                if f.endswith(ext)
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        s = np.load(self.files[i])
        x = s['x']
        # x = np.float32(x / 2**8)
        # x = np.transpose(x, (0, 3, 1, 2))

        y = s['y']
        # y = np.float32(y / 2**16)
        # y = np.transpose(y, (2, 0, 1))

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x, y