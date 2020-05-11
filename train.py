import torch
import numpy as np
import sys
import getopt
import os
import shutil
import psutil
import matplotlib.pyplot as plt
import datetime
from pytoflow.Network import TOFlow
import warnings
import ffmpeg
import pandas as pd
from dataset import SampleDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

task = 'denoising'
cuda_flag = False
pretrained_model_path = './pytoflow/toflow_models/denoise.pkl'

def show_time(now):
    s = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
        + '%02d' % now.hour + ':' + '%02d' % now.minute + ':' + '%02d' % now.second
    return s


def delta_time(datetime1, datetime2):
    if datetime1 > datetime2:
        datetime1, datetime2 = datetime2, datetime1
    second = 0
    # second += (datetime2.year - datetime1.year) * 365 * 24 * 3600
    # second += (datetime2.month - datetime1.month) * 30 * 24 * 3600
    second += (datetime2.day - datetime1.day) * 24 * 3600
    second += (datetime2.hour - datetime1.hour) * 3600
    second += (datetime2.minute - datetime1.minute) * 60
    second += (datetime2.second - datetime1.second)
    return second


def save_checkpoint(net, optimizer, epoch, losses, savepath):
    save_json = {
        'cuda_flag': net.cuda_flag,
        'h': net.height,
        'w': net.width,
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'losses': losses
    }
    torch.save(save_json, savepath)


def load_checkpoint(net, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    net.cuda_flag = checkpoint['cuda_flag']
    net.height = checkpoint['h']
    net.width = checkpoint['w']
    net.load_state_dict(checkpoint['net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    losses = checkpoint['losses']

    return net, optimizer, start_epoch, losses

def show_sample(m):
    plt.imshow(np.transpose(m, (1, 2, 0)))
    plt.axis('off')


def get_diff_norm(x, y):
    diff = y - x
    print(diff.min(), diff.max())
    diff -= diff.min()
    diff /= diff.max()
    return diff

def compare_sample(s):
    x, y = s
    x, y = x.numpy(), y.numpy()
    d = get_diff_norm(x[3], y)
    ims = [x[3], y, d]
    fig=plt.figure(figsize=(16, 16))
    columns = 2
    rows = 2
    for i in range(1, columns*rows):
        img = ims[i-1]
        fig.add_subplot(rows, columns, i)
        show_sample(img)
    fig.add_subplot(rows, columns, 4)
    plt.hist(d.flatten(), bins=20)
    plt.show()

# Hyperparams
LR = 1 * 1e-4
EPOCH = 5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1
LR_strategy = []
h = 256
w = 448

use_checkpoint = True
checkpoint_path = './checkpoints/checkpoint_1epoch.ckpt'
work_place = '.'
model_name = 'denoise_decomp'
Training_pic_path = 'Training_result.jpg'
model_information_txt = model_name + '_information.txt'

loss_func = torch.nn.L1Loss()

# Dataset
vd = SampleDataset("./data/processed_decomp")
if False:
    for i in range(3):
        compare_sample(vd[i])
        plt.savefig(f"sample_{model_name}{i}.jpg")
        plt.clf()
        plt.close()

# Train/Val/Test Splits
train_ind, val_ind, test_ind = np.split(
    np.arange(len(vd)),
    [int(.6*len(vd)), int(.8*len(vd))]
)
train_sampler = SubsetRandomSampler(train_ind)
val_sampler = SubsetRandomSampler(val_ind)
test_sampler = SubsetRandomSampler(test_ind)

# Dataloader
train_loader = DataLoader(vd, sampler=train_sampler, num_workers=4)
val_loader = DataLoader(vd, sampler=val_sampler, num_workers=4)
test_loader = DataLoader(vd, sampler=test_sampler, num_workers=4)

# Load Pretrained Model
toflow = TOFlow(h, w, task=task, cuda_flag=cuda_flag)#.cuda()
optimizer = torch.optim.Adam(toflow.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Train
prev_time = datetime.datetime.now()  # current time
print('%s  Start training...' % show_time(prev_time))
plotx = []
ploty = []
start_epoch = 0
check_loss = 1
sample_size = len(vd)

if not use_checkpoint:
    toflow.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
else:
    toflow, optimizer, start_epoch, ploty = load_checkpoint(toflow, optimizer, checkpoint_path)
    plotx = list(range(len(ploty)))
    check_loss = min(ploty) if ploty else float('inf')


for epoch in range(start_epoch, EPOCH):
    losses = 0
    count = 0
    for step, sample in enumerate(train_loader):
        print(step)
        x, y = sample
        x = x#.cuda()
        reference = y#.cuda()

        prediction = toflow(x)
        prediction = prediction#.cuda()
        loss = loss_func(prediction, reference)

        # losses += loss                # the reason why oom happened
        losses += loss.item()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        count += len(x)
        if count % 10 == 0:
            print('%s  Processed %d (%0.2f%%) triples.\tMemory used %0.2f%%.\tCpu used %0.2f%%.' %
                  (
                  show_time(datetime.datetime.now()), count, count / sample_size * 100, psutil.virtual_memory().percent,
                  psutil.cpu_percent(1)))
            print('\n%s  epoch %d: Average_loss=%f\n' % (
            show_time(datetime.datetime.now()), epoch + 1, losses / (step + 1)))

    # learning rate strategy
    if epoch in LR_strategy:
        optimizer.param_groups[0]['lr'] /= 10

    plotx.append(epoch + 1)
    ploty.append(losses / (step + 1))
    if epoch // 1 == epoch / 1:
        plt.plot(plotx, ploty)
        plt.savefig(Training_pic_path)

    # checkpoint and then prepare for the next epoch
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    save_checkpoint(toflow, optimizer, epoch + 1, ploty, './checkpoints/checkpoint_%depoch.ckpt' % (epoch + 1))

    if check_loss > losses / (step + 1):
        print('\n%s Saving the best model temporarily...' % show_time(datetime.datetime.now()))
        if not os.path.exists(os.path.join(work_place, 'toflow_models')):
            os.mkdir(os.path.join(work_place, 'toflow_models'))
        torch.save(toflow.state_dict(), os.path.join(work_place, 'toflow_models', model_name + '_best_params.pkl'))
        print('Saved.\n')
        check_loss = losses / (step + 1)

