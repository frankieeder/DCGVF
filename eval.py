from dataset import SampleDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from pytoflow.Network import TOFlow

def show_sample(m):
    plt.imshow(np.transpose(m, (1, 2, 0)))
    plt.axis('off')

def visual_results(x, y, toflow, name):
    x, y = x[0].cpu().detach().numpy(), y[0].cpu().detach().numpy()
    diff = x[3] - y
    diff_vals = diff.flatten
    diff -= diff.min()
    max_diff = diff.max()
    diff /= max_diff

    x_torch = torch.from_numpy(x[np.newaxis, ...]).cuda()
    y_hat = toflow(x_torch).cpu().detach().numpy()
    y_hat = y_hat[0]
    res = y_hat - y
    res_vals = res.flatten()
    res -= res.min()
    res /= max_diff # Scale same as original...

    w=5000
    h=5000
    fig=plt.figure(figsize=(16, 16))
    columns = 2
    rows = 5
    fig.add_subplot(rows, columns, 1)
    show_sample(x[3])
    fig.add_subplot(rows, columns, 2)
    plt.hist(x[3].flatten(), 2**12)

    fig.add_subplot(rows, columns, 3)
    show_sample(y_hat)
    fig.add_subplot(rows, columns, 4)
    plt.hist(y_hat.flatten(), 2**12)

    fig.add_subplot(rows, columns, 5)
    show_sample(res)
    fig.add_subplot(rows, columns, 6)
    plt.hist(res_vals.flatten(), 2**12)

    fig.add_subplot(rows, columns, 7)
    show_sample(y)
    fig.add_subplot(rows, columns, 8)
    plt.hist(y.flatten())

    fig.add_subplot(rows, columns, 9)
    show_sample(diff)
    fig.add_subplot(rows, columns, 10)
    plt.hist(diff_vals.flatten())


    plt.savefig(name)


# Dataset
vd = SampleDataset("../dcgvf_data/processed_decomp")

# Train/Val/Test Splits
train_ind, val_ind, test_ind = np.split(
    np.arange(len(vd)),
    [int(.6*len(vd)), int(.8*len(vd))]
)
train_sampler = SubsetRandomSampler(train_ind)
val_sampler = SubsetRandomSampler(val_ind)
test_sampler = SubsetRandomSampler(test_ind)

h = 256
w = 448
task = 'denoising'
cuda_flag = True
model_path = './toflow_models/denoise_decomp_best_params_old.pkl'

# Load Pretrained Model
toflow = TOFlow(h, w, task=task, cuda_flag=cuda_flag).cuda()
toflow.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0')))

# Dataloader
train_loader = DataLoader(vd, sampler=train_sampler, num_workers=2)
val_loader = DataLoader(vd, sampler=val_sampler, num_workers=2)
test_loader = DataLoader(vd, sampler=test_sampler, num_workers=2)

tl_iter = iter(train_loader)
for i in range(5):
    print(i)
    x, y = next(tl_iter)
    visual_results(x, y, toflow, f'train{i}result.png')

vl_iter = iter(val_loader)
for i in range(5):
    print(i)
    x, y = next(vl_iter)
    visual_results(x, y, toflow, f'val{i}result.png')

testl_iter = iter(test_loader)
for i in range(5):
    print(i)
    x, y = next(testl_iter)
    visual_results(x, y, toflow, f'test{i}result.png')

