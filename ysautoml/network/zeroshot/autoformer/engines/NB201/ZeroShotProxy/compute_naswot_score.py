'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.

The implementation of NASWOT score is modified from:
https://github.com/BayesWatch/nas-without-training

-----------
Note from the authors of AZ-NAS

The code is further modified from the implementation of ZenNAS [https://github.com/idstcv/ZenNAS/blob/d1d617e0352733d39890fb64ea758f9c85b28c1a/ZeroShotProxy/compute_NASWOT_score.py]
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

def logdet(K):
    s, ld = np.linalg.slogdet(K)
    return ld

def get_batch_jacobian(net, x):
    net.zero_grad()
    x.requires_grad_(True)
    _, y = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    # return jacob, target.detach(), y.detach()
    return jacob, y.detach()

def compute_nas_score(model, gpu, trainloader, resolution, batch_size):
    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    # network_weight_gaussian_init(model)
    init_model(model, 'kaiming_norm_fanin')
    input = torch.randn(size=[batch_size, 3, resolution, resolution])
    if gpu is not None:
        input = input.cuda(gpu)

    model.K = np.zeros((batch_size, batch_size))

    def counting_forward_hook(module, inp, out):
        try:
            if not module.visited_backwards:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()
            K = x @ x.t()
            K2 = (1. - x) @ (1. - x.t())
            model.K = model.K + K.cpu().numpy() + K2.cpu().numpy()
        except Exception as err:
            print('---- error on model : ')
            print(model)
            raise err


    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True

    for name, module in model.named_modules():
        # if 'ReLU' in str(type(module)):
        if isinstance(module, nn.ReLU):
            # hooks[name] = module.register_forward_hook(counting_hook)
            module.visited_backwards = True
            module.register_forward_hook(counting_forward_hook)
            module.register_backward_hook(counting_backward_hook)

    x = input
    jacobs, y = get_batch_jacobian(model, x)

    score = logdet(model.K)
    info = {}
    info['naswot'] = float(score)
    return info