'''
Note from the authors of AZ-NAS

We have referred to the official implementation in [https://github.com/cmu-catalyst/GradSign/blob/26412be8000a19c1a1147b2bb9399a7d1003eaf1/zero-cost-nas-code/gradsign.py].

[Major modification]
We add the condition in Ln22~23 to detect modules that are not used, instead of zeroing gradients in Ln38, which results in incorrect scores due to the "mean" operation in Ln43.
Without this condition, it could produce different results for the cases using a supernet and stand-alone networks.
'''

import os, sys
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

def get_flattened_metric(net, metric):
    grad_list = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if layer.weight.grad is None:
                continue
            grad_list.append(metric(layer).flatten())
    flattened_grad = np.concatenate(grad_list)

    return flattened_grad


def get_grad_conflict(net, inputs, targets, loss_fn=F.cross_entropy):
    N = inputs.shape[0]
    batch_grad = []
    for i in range(N):
        net.zero_grad(set_to_none=True)
        _, outputs = net.forward(inputs[[i]])
        loss = loss_fn(outputs, targets[[i]])
        loss.backward()
        flattened_grad = get_flattened_metric(net, lambda
            l: l.weight.grad.data.cpu().numpy() if l.weight.grad is not None else torch.zeros_like(l.weight).cpu().numpy())
        batch_grad.append(flattened_grad)
    batch_grad = np.stack(batch_grad)
    direction_code = np.sign(batch_grad)
    direction_code = abs(direction_code.sum(axis=0))
    score = np.nanmean(direction_code)
    # print(len(direction_code))
    return score

def compute_nas_score(network, gpu, trainloader, resolution, batch_size, batch_iter=1):
    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')
    
    network.train()
    network.to(device)

    s = []
    for i, batch in enumerate(trainloader):
        if i == batch_iter:
            break
        data,label = batch[0],batch[1]
        data,label=data.to(device),label.to(device)

        s.append(get_grad_conflict(net=network, inputs=data, targets=label, loss_fn=F.cross_entropy))
        
    score = np.mean(s)
    info = {}
    info['gradsign'] = float(score) if not np.isnan(score) else -np.inf
    return info
