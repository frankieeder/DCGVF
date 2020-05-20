from dataset import SampleDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from pytoflow.Network import TOFlow

def save_results(sample, toflow, name):
    x, y = sample
    y_hat = toflow(x.cuda())
    y_hat = y_hat.cpu().detach().numpy()
    np.savez_compressed(name, x=x, y=y, y_hat=y_hat)

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


# Saving Visual Samples
print("Saving Visual Examples...")
num_samples = 5

tl_iter = iter(train_loader)
for i in range(num_samples):
    print(i)
    s = next(tl_iter)
    save_results(s, toflow, f'train{i}result.npz')\

vl_iter = iter(val_loader)
for i in range(num_samples):
    print(i)
    s = next(vl_iter)
    save_results(s, toflow, f'val{i}result.npz')

testl_iter = iter(test_loader)
for i in range(num_samples):
    print(i)
    s = next(testl_iter)
    save_results(s, toflow, f'test{i}result.npz')

