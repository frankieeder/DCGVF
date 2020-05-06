import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import ffmpeg
import os
import numpy as np
import matplotlib.pyplot as plt

data_root = '/Volumes/Elements/Camera Obscura Backup Footage/'
all_files = []
for root, folders, files in os.walk(data_root):
    all_files.extend([os.path.join(root, f) for f in files if f.endswith('001.mov')])
#print(sum([root + f for f in files if f.endswith('.R3D')] for root, folders, files in os.walk(data_root)))
print(all_files[0])
height, width = 1322, 1102
info = ffmpeg.probe(all_files[0])
w, h = info['streams'][0]['width'], info['streams'][0]['height']

x=2
'''out, _ = (
    ffmpeg
    .input(all_files[0])
    .filter('select', 'gte(n,{})'.format(1))
    .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
    .run(capture_stdout=True)
)'''
'''out, err = (
    ffmpeg
    .input(all_files[0])
    .output('pipe:', format='rawvideo', pix_fmt='rgb24', vf="select='between(n\,0\,7)'")
    .run(capture_stdout=True)#, '-b:v': '200k'})
)
x=2
out, err = (
    ffmpeg
    .input(out)
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')#, '-b:v': '200k'})
    .run(capture_stdout=True)
)
#print(out)
x=2
video_h264 = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
x=2'''
in_file = ffmpeg.input(all_files[0])#, vf="select='between(n\,0\,7)'", vsync=0)
video_prores_stream, err = in_file.output(
    'pipe:',
    format='rawvideo',
    pix_fmt='rgb48',
    vf="select='between(n\,0\,7)'",
    vsync=0
).run(capture_stdout=True)
video_prores = np.frombuffer(video_prores_stream, np.uint16).reshape([-1, h, w, 3])
x=2
video_h264_stream = (
    ffmpeg.input(all_files[0])
    .output('pipe:', format='rawvideo', pix_fmt='rgb24', vcodec='h264')
    .input('pipe:')
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
)
video_h264 = np.frombuffer(video_h264_stream.run(), np.uint8).reshape([-1, h, w, 3])
'''out, err = (
    ffmpeg
    
    .output('pipe:', format='rawvideo', pix_fmt='rgb48', vf="select='between(n\,0\,7)'", vsync=0)
    .run(capture_stdout=True)
)'''
video_prores = np.frombuffer(out, np.uint16).reshape([-1, h, w, 3])
a = (video_prores / (2**16)) - (video_h264 / (2**8))
a -= a.min()
a /= a.max()
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




