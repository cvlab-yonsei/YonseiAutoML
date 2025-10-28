'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.

-----------
Note from the authors of AZ-NAS

The code is modified from the implementation of ZenNAS [https://github.com/idstcv/ZenNAS/blob/d1d617e0352733d39890fb64ea758f9c85b28c1a/ZeroShotProxy/compute_gradnorm_score.py]

We revise the code as follows:
1. Make it compatible with NAS-Bench-201
2. Initialize the model with Kaiming init
'''


import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np

# def network_weight_gaussian_init(net: nn.Module):
#     with torch.no_grad():
#         for m in net.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight)
#                 if hasattr(m, 'bias') and m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight)
#                 if hasattr(m, 'bias') and m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             else:
#                 continue

#     return net

def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    else:
        raise NotImplementedError
    return model

import torch.nn.functional as F
def cross_entropy(logit, target):
    # target must be one-hot format!!
    prob_logit = F.log_softmax(logit, dim=1)
    loss = -(target * prob_logit).sum(dim=1).mean()
    return loss

def compute_nas_score(model, gpu, trainloader, resolution, batch_size):

    model.train()
    model.requires_grad_(True)

    model.zero_grad()

    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    # network_weight_gaussian_init(model)
    init_model(model, 'kaiming_norm_fanin')
    input = torch.randn(size=[batch_size, 3, resolution, resolution])
    if gpu is not None:
        input = input.cuda(gpu)
    _, output = model(input)
    # y_true = torch.rand(size=[batch_size, output.shape[1]], device=torch.device('cuda:{}'.format(gpu))) + 1e-10
    # y_true = y_true / torch.sum(y_true, dim=1, keepdim=True)

    num_classes = output.size(1)
    y = torch.randint(low=0, high=num_classes, size=[batch_size])

    one_hot_y = F.one_hot(y, num_classes).float()
    if gpu is not None:
        one_hot_y = one_hot_y.cuda(gpu)

    loss = cross_entropy(output, one_hot_y)
    loss.backward()
    norm2_sum = 0
    with torch.no_grad():
        for p in model.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                norm2_sum += torch.norm(p.grad) ** 2

    grad_norm = float(torch.sqrt(norm2_sum))
    info = {}
    info['grad_norm'] = grad_norm
    return info


