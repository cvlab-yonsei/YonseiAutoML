import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd.function import InplaceFunction
from scipy.stats import norm
import math
import functools

class STE_round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


def scale_gradient(x, scale):
    return (x - x * scale).detach() + x * scale

class QuantConv(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                bitW=8,
                bitA=8,
                ):
        super(QuantConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask = nn.Parameter(torch.ones_like(self.weight), requires_grad=False)
        self.mask.requires_grad = False
        self.bitW = bitW
        self.bitA = bitA
        self.s_w = nn.Parameter(data = torch.tensor(1.))
        self.s_x, self.z_x = nn.Parameter(data = torch.tensor(1.)), nn.Parameter(data = torch.tensor(0.), requires_grad = False)
        self.round_ste = STE_round.apply

        self.inited = False
        self.disable_act_quant = False

    @torch.no_grad()
    def init_quantization_scale(self, x, w):
        w_l, w_u = torch.quantile(w, 1e-5), torch.quantile(w, 1-1e-5)
        w_o = torch.maximum(w_l, w_u)
        s_w = w_o / (2 ** (self.bitW - 1) - 1)
        self.s_w.fill_(s_w)

        if not self.disable_act_quant:
            if x.numel() > 1000000:
                from scipy.stats import norm
                sample_ratio = 1000000/x.numel()
                x_sample = torch.randn_like(x) > float(norm.ppf(1 - sample_ratio))
                x_sample = x[x_sample]
            else:
                x_sample = x

            x_l, x_u = torch.quantile(x_sample, 1e-5), torch.quantile(x_sample, 1-1e-5)
            s_x = (x_u - x_l) / (2 ** self.bitA - 1)
            z_x = torch.clamp((-x_l / s_x).round(), 0, 2 ** self.bitA - 1).item()
            self.s_x.fill_(s_x)
            self.z_x.fill_(z_x)


    def forward(self, x):
        if self.inited is False:
            self.init_quantization_scale(x, self.weight)
            self.inited = True

        s_w = self.s_w
        s_x = self.s_x
        weight_int = self.round_ste(self.weight / s_w)
        weight_int = torch.clamp(weight_int, -2 ** (self.bitW - 1), 2 ** (self.bitW - 1) - 1)
        weight_dequant = weight_int * s_w

        # Activation quantization (zero-point quantization)
        x_int = self.round_ste(x / s_x) + self.z_x
        x_int = torch.clamp(x_int, 0, 2 ** self.bitA - 1)
        x_dequant = (x_int - self.z_x) * s_x
        out = F.conv2d(x_dequant, weight_dequant * self.mask, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        return out

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def quantize_model(model, n_bits):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            new_conv = QuantConv(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias is not None, bitW = n_bits, bitA = n_bits)

            new_conv.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_conv.bias.data.copy_(module.bias.data)
            rsetattr(model, name, new_conv)


    return model