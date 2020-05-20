import torch
from torch.utils.data import Dataset
import numpy as np
import os

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

class HistEqualizedSampleDataset(SampleDataset):
    """Basic Dataset with Histogram Equalization"""
    def __getitem__(self, i):
        x, y = super().__getitem__(i)

        T = self.HistEqFunctionCreate(y)

        x = self.HistEqFunctionApply(T, x)
        y = self.HistEqFunctionApply(T, y)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x, y


    def HistEqFunctionCreate(self, y):
        pdf, bins = np.histogram(y.flatten(), bins=2**16, normed=True)
        pdf = pdf / pdf.sum()
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf.max()
        return cdf

    def HistEqFunctionApply(self, T, m):
        m_inds = (m*(2**16)).int()
        m_T = T[m_inds]
        m_T = (m_T / (2**16)).astype(float)
        return m_T

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def show_sample(m):
        plt.imshow(np.transpose(m, (1, 2, 0)))


    def get_diff_norm(x, y):
        diff = y - x
        print(diff.min(), diff.max())
        diff -= diff.min()
        diff /= diff.max()
        return diff


    def compare_sample(s):
        x, y = s
        x, y = x.numpy(), y.numpy()
        ims = [x[3], y, get_diff_norm(x[3], y)]
        w = 5000
        h = 5000
        fig = plt.figure(figsize=(16, 16))
        columns = 1
        rows = 3
        for i in range(1, columns * rows + 1):
            img = ims[i - 1]
            fig.add_subplot(rows, columns, i)
            show_sample(img)
        plt.show()

    vd = SampleDataset('/Volumes/Elements/dcgvf_data/processed_decomp')
    #compare_sample(vd[0])

    vd_eq = HistEqualizedSampleDataset('/Volumes/Elements/dcgvf_data/processed_decomp')
    compare_sample(vd_eq[0])

    x=2