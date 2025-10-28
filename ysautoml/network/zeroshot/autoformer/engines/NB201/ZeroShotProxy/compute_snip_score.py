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

The code is modified from the implementation of zero-cost-nas [https://github.com/SamsungLabs/zero-cost-nas/blob/a43bfbc90c1a02a81fb4397a5e096759e93fbe50/foresight/pruners/measures/snip.py]

We revise the code as follows:
1. Make it compatible with NAS-Bench-201
2. Wrap the function for computing snip score with compute_nas_score
3. Initialize the model with Kaiming init
'''


import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np

import types
import torch.nn.functional as F

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

def snip_forward_conv2d(self, x):
    return F.conv2d(
        x,
        self.weight * self.weight_mask,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )


def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)


def compute_snip_per_weight(net, inputs, targets, mode, loss_fn, split_data=1):
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        _, outputs = net.forward(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

    # select the gradients that we want to use for search/prune
    def snip(layer):
        if layer.weight_mask.grad is not None:
            return torch.abs(layer.weight_mask.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, snip, mode)

    return grads_abs

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

    grads_abs_list = compute_snip_per_weight(net=model, inputs=input, targets=target, mode='', loss_fn=torch.nn.CrossEntropyLoss())
    score = 0
    for grad_abs in grads_abs_list:
        score = score + grad_abs.sum().item()

    info = {}
    info['snip'] = score

    return info


