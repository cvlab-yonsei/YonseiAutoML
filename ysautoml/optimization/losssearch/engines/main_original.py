import argparse
import logging
import os
import random

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch.nn as nn

from model import *
from loss import *

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()


valid_size = 0.95
random_seed = 1
noise_ratio = 0.2
dataset_path = '/dataset'
train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# load the dataset
LS_train_dataset = torchvision.datasets.CIFAR10(
    root=dataset_path, train=True,
    download=True, transform=train_transform,
)
LS_valid_dataset = torchvision.datasets.CIFAR10(
    root=dataset_path, train=True,
    download=True, transform=valid_transform,
)
train_dataset = torchvision.datasets.CIFAR10(
    root=dataset_path, train=True,
    download=True, transform=train_transform,
)
test_dataset = torchvision.datasets.CIFAR10(
    root=dataset_path, train=False,
    download=True, transform=valid_transform
)
num_train = len(LS_train_dataset) # 50k
indices = list(range(num_train))
split = int(valid_size * num_train) # 0.5 * 50k = 25k
# shuffle
np.random.seed(random_seed)
np.random.shuffle(indices)
train_idx, val_idx = indices[:split], indices[split:]
LS_train_set = torch.utils.data.Subset(LS_train_dataset, train_idx)
LS_val_set = torch.utils.data.Subset(LS_valid_dataset, val_idx)
# add noise
for i in range(int(noise_ratio * split)): # 0.2 * 25k = 5k
    data_idx = train_idx[i]
    origin_target = LS_train_set.dataset.targets[data_idx]
    targets = list(range(10))
    targets.remove(origin_target)
    LS_train_set.dataset.targets[data_idx] = np.random.choice(targets)
np.random.shuffle(indices)
for i in range(int(noise_ratio * len(train_dataset.targets))): # 0.2 * 50k = 10k
    data_idx = indices[i]
    origin_target = train_dataset.targets[data_idx]
    targets = list(range(10))
    targets.remove(origin_target)
    train_dataset.targets[data_idx] = np.random.choice(targets)
    
    
LS_train_loader = torch.utils.data.DataLoader(LS_train_set, batch_size=128,
                                            shuffle=True, num_workers=2)
LS_val_loader = torch.utils.data.DataLoader(LS_val_set, batch_size=128,
                                            shuffle=True, num_workers=2)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                            shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                         shuffle=False, num_workers=2)


model = resnet20().cuda()

optimizer_model = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)




criterion = custom_loss().cuda()

optimizer_loss = torch.optim.SGD(criterion.parameters(), lr=0.0001, momentum = 0.9)
scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_model, T_max=args.epochs, eta_min=0)

scheduler_loss = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_loss, T_max=args.epochs, eta_min=0)

for ep in range(args.epochs):
    model.train()
    correct_classified = 0
    total = 0
    for i, (images, labels) in enumerate(LS_train_loader):
        images = images.cuda()
        labels = labels.cuda()
        
        optimizer_model.zero_grad()
        optimizer_loss.zero_grad()
            
        pred = model(images)
        
        _, predicted = torch.max(pred.data, 1)
        total += pred.size(0)
        correct_classified += (predicted == labels).sum().item()
        loss= criterion(pred, labels)
        
        loss.backward()
        
        optimizer_model.step()       
#         optimizer_loss.step()
    train_acc = correct_classified/total*100
    if ep % 10 == 0 and ep != 0:
        for i, (images, labels) in enumerate(LS_val_loader):
            images = images.cuda()
            labels = labels.cuda()

            optimizer_model.zero_grad()
            optimizer_loss.zero_grad()

            pred = model(images)
            loss = criterion(pred, labels)

            loss.backward()


            optimizer_loss.step()
        print(f'Update Theta --> theta: {criterion.theta}')
        
    scheduler_model.step()
    scheduler_loss.step()
    
    with torch.no_grad():
        model.eval()
        correct_classified = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            total += pred.size(0)
            correct_classified += (predicted == labels).sum().item()
        test_acc = correct_classified/total*100
    
    print(f'ep: {ep}, train_acc: {train_acc}, test_acc: {test_acc}')