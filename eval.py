from dataset import SampleDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from pytoflow.Network import TOFlow
import pytorch_ssim

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
model_path = './toflow_models/denoise_decomp_best_params.pkl'

# Load Pretrained Model
toflow = TOFlow(h, w, task=task, cuda_flag=cuda_flag).cuda()
toflow.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0')))

# Dataloader
train_loader = DataLoader(vd, sampler=train_sampler, num_workers=2)
val_loader = DataLoader(vd, sampler=val_sampler, num_workers=2)
test_loader = DataLoader(vd, sampler=test_sampler, num_workers=2)


# Saving Visual Samples
print("Saving Visual Examples...")

tl_iter = iter(train_loader)
for i in range(5):
    print(i)
    s = next(tl_iter)
    save_results(s, toflow, f'train{i}result.npz')\

vl_iter = iter(val_loader)
for i in range(5):
    print(i)
    s = next(vl_iter)
    save_results(s, toflow, f'val{i}result.npz')

testl_iter = iter(test_loader)
for i in range(20):
    print(i)
    s = next(testl_iter)
    save_results(s, toflow, f'test{i}result.npz')

L2loss = torch.nn.MSELoss()
L1loss = torch.nn.L1Loss()
SSIMloss = pytorch_ssim.SSIM()

totalL1Loss = 0
totalL2Loss = 0
totalL1std = 0
totalL2std = 0
totalGTL1Loss = 0
totalGTL2Loss = 0
totalSSIMLoss = 0

total10BitAcc = 0
total10BitPrec = 0
total11BitAcc = 0
total11BitPrec = 0
total12BitAcc = 0
total12BitPrec = 0

def bitThresholdPercentage(L1, bd):
    thresh = 2**(-bd)
    return (L1 < thresh).int().sum().float() / L1.numel()

def bitBucketPercentage(L1, bd):
    buckets = torch.fmod((L1*(2**bd)).int(), 2**(bd-8))
    bucketPerc = (buckets != 0).int().sum().float() / L1.numel()
    return bucketPerc

for i, sample in enumerate(test_loader):
    i = i+1
    print(f"Computing sample {i}/{len(test_ind)}")
    x, y = sample
    x = x.cuda()
    print(f"{x.min(), x.max()}")
    X = x.clone()
    reference = y.cuda()

    prediction = toflow(X)
    prediction = prediction.cuda()

    with torch.no_grad():
        thisSSIMLoss = SSIMloss(prediction, reference)
        totalSSIMLoss += thisSSIMLoss.item()

        res = prediction - reference
        #this_L2loss = L2loss(prediction, reference)
        L2 = (res ** 2)
        totalL2Loss += L2.mean() #this_L2loss.item()
        totalL2std += L2.std()

        #this_L1loss = L1loss(prediction, reference)
        L1 = torch.abs(res)
        totalL1Loss += L1.mean() #this_L1loss.item()
        totalL1std += L1.std()

        total10BitAcc += bitThresholdPercentage(L1, 10)
        total10BitPrec += bitBucketPercentage(L1, 10)
        total11BitAcc += bitThresholdPercentage(L1, 11)
        total11BitPrec += bitBucketPercentage(L1, 11)
        total12BitAcc += bitThresholdPercentage(L1, 12)
        total12BitPrec += bitBucketPercentage(L1, 12)

        #print(x[:, 3].shape)
        #print(reference.shape)
        
        diff = x[:,3] - reference
        #this_GTL2loss = L2loss(x[:, 3], reference)
        totalGTL2Loss += (diff**2).mean() #this_GTL2loss.item()

        #this_GTL1loss = L1loss(x[:, 3], reference)
        totalGTL1Loss += torch.abs(diff).mean() #this_GTL1loss.item()
    #print(f"Ranges x:{x.min(), x.max()}, y:[{y.min(), y.max()}, yp: {prediction.min(), prediction.max()}")
    print(f"Avg L2 Loss: {totalL2Loss / i}, Avg L2 Loss Std: {totalL2std / i}")
    print(f"Avg L1 Loss: {totalL1Loss / i}, Avg L1 Loss Std: {totalL1std / i}")
    print(f"Avg SSIM: {totalSSIMLoss / i}")
    print(f"Avg 10 Bit Acc, Prec: {total10BitAcc / i}, {total10BitPrec / i}")
    print(f"Avg 11 Bit Acc, Prec: {total11BitAcc / i}, {total11BitPrec / i}")
    print(f"Avg 12 Bit Acc, Prec: {total12BitAcc / i}, {total12BitPrec / i}")
    print(f"Avg GT L2 Loss: {totalGTL2Loss / i}, Avg GT L1 Loss: {totalGTL1Loss / i}")
