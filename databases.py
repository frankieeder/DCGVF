import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import ffmpeg
import os
import numpy as np

data_root = '/Volumes/Elements/Camera Obscura Backup Footage/'
all_files = []
for root, folders, files in os.walk(data_root):
    all_files.extend([root + f for f in files if f.endswith('.R3D')])
#print(sum([root + f for f in files if f.endswith('.R3D')] for root, folders, files in os.walk(data_root)))
print(all_files[0])

out, err = (
    ffmpeg
    .input(all_files[0])
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run(capture_stdout=True)
)
print(out)
#class R3DVideoDataset(Dataset):




