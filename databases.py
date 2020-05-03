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
    all_files.extend([os.path.join(root, f) for f in files if f.endswith('001.mov')])
#print(sum([root + f for f in files if f.endswith('.R3D')] for root, folders, files in os.walk(data_root)))
print(all_files[0])
height, width = 720, 864
info = ffmpeg.probe(all_files[0])
out, err = (
    ffmpeg
    .input(all_files[0])
    .output('pipe:', format='rawvideo', pix_fmt='rgb48', vf="select='between(n\,0\,7)'", vsync=0)
    .run(capture_stdout=True)
)
#print(out)
video = np.frombuffer(out, np.uint16).reshape([-1, height, width, 3])
print(video.shape)

print(video.min(), video.max())
#class R3DVideoDataset(Dataset):


class VideoFrame(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample




