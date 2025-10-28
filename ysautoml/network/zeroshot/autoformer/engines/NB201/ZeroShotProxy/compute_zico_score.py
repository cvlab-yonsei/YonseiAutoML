'''
Note from the authors of AZ-NAS

The code is modified from the implementation of ZiCo [https://github.com/SLDGroup/ZiCo/blob/b0fec65923a90e84501593f675b1e2f422d79e3d/ZeroShotProxy/compute_zico.py]

We revise the code as follows:
1. Make it compatible with NAS-Bench-201
2. Check whether gradients are valid or not
3. Remove redundant variables
4. Change the name of the function
'''

import os, sys
import torch
from torch import nn
import numpy as np

def getgrad(model:torch.nn.Module, grad_dict:dict, step_iter=0):
    if step_iter==0:
        for name,mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                if mod.weight.grad is not None:
                    # print(mod.weight.grad.data.size())
                    # print(mod.weight.data.size())
                    grad_dict[name]=[mod.weight.grad.data.cpu().reshape(-1).numpy()]
    else:
        for name,mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                if mod.weight.grad is not None:
                    grad_dict[name].append(mod.weight.grad.data.cpu().reshape( -1).numpy())
    return grad_dict

def caculate_zico(grad_dict):
    allgrad_array=None
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname]= np.array(grad_dict[modname])
    nsr_mean_sum = 0
    nsr_mean_sum_abs = 0
    nsr_mean_avg = 0
    nsr_mean_avg_abs = 0
    for j, modname in enumerate(grad_dict.keys()):
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx])
        if tmpsum==0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
            nsr_mean_avg_abs += np.log(np.mean(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx]))
    return nsr_mean_sum_abs

def compute_nas_score(network, gpu, trainloader, resolution, batch_size, batch_iter=2):
    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')
    
    grad_dict= {}
    network.train()
    lossfunc = nn.CrossEntropyLoss().cuda()

    network.to(device)
    for i, batch in enumerate(trainloader):
        if i == batch_iter:
            break
        network.zero_grad(set_to_none=True)
        data,label = batch[0],batch[1]
        data,label=data.to(device),label.to(device)

        _, logits = network(data)
        loss = lossfunc(logits, label)
        loss.backward()
        grad_dict= getgrad(network, grad_dict,i)
        
    res = caculate_zico(grad_dict)
    info = {}
    info['zico'] = res
    return info
