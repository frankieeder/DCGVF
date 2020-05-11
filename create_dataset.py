data_folder = "/Volumes/Elements/Camera Obscura Backup Footage/"

import numpy as np
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import datetime
import ffmpeg

class BDIPreProcessor:
    """Creates a dataset by randomly sampling the input files."""

    def __init__(self, data_folder, out_folder, output_size, size_targ, stride,
                 reference_ind, ns_retrieval,
                 search_suffix="", files_per_folder=50, start_count=0):
        self.data_folder = data_folder
        self.out_folder = out_folder
        self.output_size = output_size
        self.size_targ = size_targ
        self.stride = stride
        self.reference_ind = reference_ind
        self.ns_retrieval = ns_retrieval
        self.files_per_folder = files_per_folder
        self.count = start_count

        self.all_files = []
        for root, folders, files in os.walk(self.data_folder):
            self.all_files.extend([
                osp.join(root, f)
                for f
                in files
                if f.endswith(search_suffix)
            ])
        print(self.all_files)

        self.info = []
        failed = []
        for i, f in enumerate(self.all_files):
            try:
                self.info.append(ffmpeg.probe(f))
                print(f"Successfully Collected Meta for file {i}: {f}")
            except ffmpeg.Error as e:
                print('stdout:', e.stdout.decode('utf8'))
                print('stderr:', e.stderr.decode('utf8'))
                print("Failed to collect data. Removing from Preprocessing...")
                failed.append(i)
        self.all_files = [f for i, f in enumerate(self.all_files) if i not in failed]
        self.info = [f['streams'][0] for f in self.info]

        self.num_samples = np.array([max(int(f['nb_frames']) - stride, 0) for f in self.info])
        self.file_probs = self.num_samples / self.num_samples.sum()

    def process(self):
        out_format = os.path.join(self.out_folder, )
        # print(out_format)
        samples_left = self.size_targ - self.count
        for _ in range(int(samples_left // self.ns_retrieval)):
            sample = self.get_sample_full_frame()
            for _ in range(self.ns_retrieval):
                print(self.count)
                rc = self.random_crop(sample)
                self.save_sample(rc, self.count)
                self.count += 1

    def save_sample(self, sample, count):
        file_format = "{}.npz".format(count)
        folder_num = int(count // self.files_per_folder)
        # print(folder_num)
        folder_dir = osp.join(self.out_folder, str(folder_num))
        # print(folder_dir)
        if not osp.isdir(folder_dir):
            os.mkdir(folder_dir)
        file_dir = osp.join(folder_dir, file_format)
        # print(file_dir)
        np.savez_compressed(file_dir, **sample)

    def get_sample_full_frame(self):
        vid_num = np.random.choice(np.arange(len(self.all_files)), p=self.file_probs)
        vid_dir = self.all_files[vid_num]
        start_frame = np.random.randint(0, self.num_samples[vid_num])

        print(vid_num, vid_dir, start_frame, self.stride)

        y_frames = self.read_video(vid_num, vid_dir, start_frame, self.stride, pix_fmt='rgb48', dtype=np.uint16)
        y_frame = y_frames[self.reference_ind]

        x_frames = (y_frames / 2 ** 8).astype(np.uint8)

        # Adjust X Frames
        x_bit_depth = 2 ** (8 * x_frames.itemsize)
        x_frames = np.float32(x_frames / x_bit_depth)
        x_frames = np.transpose(x_frames, (0, 3, 1, 2))

        # Adjust y Frame
        y_bit_depth = 2 ** (8 * y_frame.itemsize)
        y_frame = np.float32(y_frame / y_bit_depth)
        y_frame = np.transpose(y_frame, (2, 0, 1))

        return {'x': x_frames, 'y': y_frame}

    def random_crop(self, s):
        """sample is a dict of matrices of shape ([f,] c, h, w). We apply the same crop to all."""
        # print(s['x'].shape, s['y'].shape)
        h, w = list(s.values())[0].shape[-2:]
        # print(h, w)
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        new_sample = {}
        for k, v in s.items():
            new_sample[k] = v[..., top:top + new_h, left:left + new_w]
            # print(new_sample[k].shape, new_sample[k].min(), new_sample[k].max(), np.histogram(new_sample[k].flatten()))
        return new_sample

    def get_meta(self, vid_num):
        # Meta using cached info
        vid_meta = self.info[vid_num]
        total_frames = int(vid_meta['nb_frames'])
        w, h = vid_meta['width'], vid_meta['height']
        framerate_str = vid_meta['avg_frame_rate']
        num, denom = framerate_str.split("/")
        framerate = int(num) / int(denom)
        return w, h, framerate

    def read_video(self, vid_num, vid_dir, start_frame, num_frames, pix_fmt='rgb48', dtype=np.uint16):
        w, h, framerate = self.get_meta(vid_num)
        # Pull Data
        try:
            print("Pulling Video Chunk...")
            vid_stream, err = (
                ffmpeg
                    .input(vid_dir, ss=start_frame / framerate)
                    .output(
                    'pipe:',
                    format='rawvideo',
                    pix_fmt=pix_fmt,
                    vframes=num_frames,
                    vsync=0
                )
                    .run(capture_stdout=True, capture_stderr=True)
            )
            vid = np.frombuffer(vid_stream, dtype).reshape([-1, h, w, 3])
            return vid
        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
            raise e


class DecompressionPreProcessor(BDIPreProcessor):
    def get_sample_full_frame(self):
        vid_num = np.random.choice(np.arange(len(self.all_files)), p=self.file_probs)
        vid_dir = self.all_files[vid_num]
        start_frame = np.random.randint(0, self.num_samples[vid_num])

        y_frames = self.read_video(vid_num, vid_dir, start_frame, self.stride, pix_fmt='rgb48', dtype=np.uint16)
        y_frame = y_frames[self.reference_ind]

        temp_dir = './temp.mp4'
        self.convert_video(vid_num, temp_dir, start_frame, self.stride)
        x_frames = self.read_video(vid_num, temp_dir, 0, self.stride, pix_fmt='rgb24', dtype=np.uint8)

        # Adjust X Frames
        x_bit_depth = 2 ** (8 * x_frames.itemsize)
        x_frames = np.float32(x_frames / x_bit_depth)
        x_frames = np.transpose(x_frames, (0, 3, 1, 2))

        # Adjust y Frame
        y_bit_depth = 2 ** (8 * y_frame.itemsize)
        y_frame = np.float32(y_frame / y_bit_depth)
        y_frame = np.transpose(y_frame, (2, 0, 1))

        return {'x': x_frames, 'y': y_frame}

    def convert_video(self, vid_num, out_fn, start_frame, num_frames):
        w, h, framerate = self.get_meta(vid_num)
        vid_dir = self.all_files[vid_num]
        # Pick random bitrate
        br = np.random.randint(5e5, 3e6)
        print("Converting to mp4 and caching...")
        try:
            (
                ffmpeg
                    .input(vid_dir, ss=start_frame / framerate)
                    .output(out_fn, vcodec='h264', vb=br, vframes=num_frames)
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
            raise e

decomp_proc = DecompressionPreProcessor(
    data_folder,
    "./data/processed_decomp",
    output_size=(256, 448),
    stride=7,
    reference_ind=3,
    size_targ=1e5,
    ns_retrieval=50,
    search_suffix='_001.mov'
)
decomp_proc.process()


bdi_proc = BDIPreProcessor(
    data_folder,
    "./data/processed_bdi",
    output_size = (256, 448),
    stride = 7,
    reference_ind = 3,
    size_targ = 1e5,
    ns_retrieval = 50,
    search_suffix='_001.mov',
    start_count = 12000
)
bdi_proc.process()


