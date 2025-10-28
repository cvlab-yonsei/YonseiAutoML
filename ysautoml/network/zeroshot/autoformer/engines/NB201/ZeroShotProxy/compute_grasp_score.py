# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
'''
Note from the authors of AZ-NAS

The code is modified from the implementation of zero-cost-nas [https://github.com/SamsungLabs/zero-cost-nas/blob/a43bfbc90c1a02a81fb4397a5e096759e93fbe50/foresight/pruners/measures/grasp.py]

We revise the code as follows:
1. Make it compatible with NAS-Bench-201
2. Wrap the function for computing grasp score with compute_nas_score
3. Initialize the model with Kaiming init
'''

import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np

import torch.autograd as autograd

import torch

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

def get_layer_metric_array(net, metric, mode):
    metric_array = []

    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))

    return metric_array

def compute_grasp_per_weight(net,
                             inputs,
                             targets,
                             mode,
                             loss_fn,
                             T=1,
                             num_iters=1,
                             split_data=1):

    # get all applicable weights
    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)
            layer.weight.requires_grad_(True)  # TODO isn't this already true?

    # NOTE original code had some input/target splitting into 2
    # I am guessing this was because of GPU mem limit
    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        # forward/grad pass # 1
        grad_w = None
        for _ in range(num_iters):
            # TODO get new data, otherwise num_iters is useless!
            _, outputs = net.forward(inputs[st:en])
            outputs = outputs / T 
            loss = loss_fn(outputs, targets[st:en])
            grad_w_p = autograd.grad(loss, weights, allow_unused=True)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        # forward/grad pass # 2
        _, outputs = net.forward(inputs[st:en])
        outputs = outputs / T
        loss = loss_fn(outputs, targets[st:en])
        grad_f = autograd.grad(
            loss, weights, create_graph=True, allow_unused=True)

        # accumulate gradients computed in previous step and call backwards
        z, count = 0, 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if grad_w[count] is not None:
                    z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

    # compute final sensitivity metric and put in grads
    def grasp(layer):
        if layer.weight.grad is not None:
            return -layer.weight.data * layer.weight.grad  # -theta_q Hg
            # NOTE in the grasp code they take the *bottom* (1-p)% of values
            # but we take the *top* (1-p)%, therefore we remove the -ve sign
            # EDIT accuracy seems to be negatively correlated with this metric,
            #  so we add -ve sign here!
        else:
            return torch.zeros_like(layer.weight)

    grads = get_layer_metric_array(net, grasp, mode)

    return grads

def compute_nas_score(model, gpu, trainloader, resolution, batch_size):
    model.train()
    model.requires_grad_(True)

    model.zero_grad()

    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    # network_weight_gaussian_init(model)
    init_model(model, 'kaiming_norm_fanin')
    input, target = next(iter(trainloader))

    if gpu is not None:
        input = input.cuda(gpu)
        target = target.cuda(gpu)

    grads_list = compute_grasp_per_weight(net=model, inputs=input, targets=target, mode='', loss_fn=torch.nn.CrossEntropyLoss())
    score = 0
    for grad in grads_list:
        score = score + grad.sum().item()

    info = {}
    info['grasp'] = score

    return info


