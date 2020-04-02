import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import ffmpeg
import os
import numpy as np

data_root = '/Volumes/Backup_Elem/Footage_Proxies_Old/'
all_files = []
for root, folders, files in os.walk(data_root):
    all_files.extend([os.path.join(root, f) for f in files if f.endswith('.mp4')])
#print(sum([root + f for f in files if f.endswith('.R3D')] for root, folders, files in os.walk(data_root)))
print(all_files[0])

out, err = (
    ffmpeg
    .input(all_files[0])
    .output('pipe:', format='rawvideo', pix_fmt='rgb48', vf="select='between(n\,0\,7)'", vsync=0)
    .run(capture_stdout=True)
)
#print(out)
video = np.frombuffer(out, np.uint16).reshape([-1, 720, 864, 3])
print(video.shape)
print(video.min(), video.max())
#class R3DVideoDataset(Dataset):




