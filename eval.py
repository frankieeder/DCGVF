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
    diff -= diff.min()
    max_diff = diff.max()
    diff /= max_diff

    x_torch = torch.from_numpy(x[np.newaxis, ...]).cuda()
    y_hat = toflow(x_torch).cpu().detach().numpy()
    y_hat = y_hat[0]
    res = y_hat - y
    res -= res.min()
    res /= max_diff # Scale same as original...

    ims = [x[3], y_hat, y, diff, res]

    w=5000
    h=5000
    fig=plt.figure(figsize=(16, 16))
    columns = 2
    rows = 5
    for i in range(1, rows +1):
        img = ims[i-1]
        fig.add_subplot(rows, columns, i)
        show_sample(img)
        fig.add_subplot(rows, columns, i+1)
        plt.hist(img.flatten())
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

