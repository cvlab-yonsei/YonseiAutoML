#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
import pdb
import numpy as np
from collections import OrderedDict

class Quantizer_w_scale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bit, scale):
        if bit == 1:
            s = (2 ** bit) - 1
        else:
            s = (2 ** bit) - 2

        out_f = x * s
        out_r = out_f.round()

        diff = out_f - out_r

        ctx.save_for_backward(diff, s, bit, scale)

        return out_r / s
    @staticmethod
    def backward(ctx, g):
        diff, s, bit, scale = ctx.saved_tensors
        grad_b = (diff / (s**2)) * (2**bit) * torch.log(torch.tensor(2.0))
        grad_b_norm = grad_b / (grad_b.norm(1) / len(grad_b.reshape(-1)))

        torch.max(grad_b.abs())

        return g * scale, grad_b_norm * g, None

class Quantizer_a_scale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bit, scale):
        s = (2 ** bit) - 1

        out_f = x * s
        out_r = out_f.round()

        diff = out_f - out_r

        ctx.save_for_backward(diff, s, bit, scale)

        return out_r / s
    @staticmethod
    def backward(ctx, g):
        diff, s, bit, scale = ctx.saved_tensors
        grad_b = (diff / (s**2)) * (2**bit) * torch.log(torch.tensor(2.0))
        # grad_b_norm = grad_b / torch.max(grad_b.abs())
        grad_b_norm = grad_b / (grad_b.norm(1) / len(grad_b.reshape(-1)))
        return g * scale, grad_b_norm * g, None

class Quantizer_w(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bit):
        if bit == 1:
            s = (2 ** bit) - 1
        else:
            s = (2 ** bit) - 2

        out_f = x * s
        out_r = out_f.round()

        diff = out_f - out_r

        ctx.save_for_backward(diff, s, bit)

        return out_r / s
    @staticmethod
    def backward(ctx, g):
        diff, s, bit = ctx.saved_tensors
        grad_b = (diff / (s**2)) * (2**bit) * torch.log(torch.tensor(2.0))
        grad_b_norm = grad_b / (grad_b.norm(1) / len(grad_b.reshape(-1)))
        
        torch.max(grad_b.abs())

        return g, grad_b_norm * g

class Quantizer_a(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bit):
        s = (2 ** bit) - 1

        out_f = x * s
        out_r = out_f.round()

        diff = out_f - out_r

        ctx.save_for_backward(diff, s, bit)

        return out_r / s
    @staticmethod
    def backward(ctx, g):
        diff, s, bit = ctx.saved_tensors
        grad_b = (diff / (s**2)) * (2**bit) * torch.log(torch.tensor(2.0))
        # grad_b_norm = grad_b / torch.max(grad_b.abs())
        grad_b_norm = grad_b / (grad_b.norm(1) / len(grad_b.reshape(-1)))
        return g, grad_b_norm * g

class q_transform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, T):
        x = x - 0.5
        coeff = torch.tanh(0.5 * T)
        out = x - torch.round(x)
        out = torch.tanh(T * out)
        out = out / coeff
        out = (out + 1) * 0.5

        out = out + torch.round(x)
        return out
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class RoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n):
        return torch.round(x * n) / n
    @staticmethod
    def backward(ctx, grad_output):
        # grad_input = grad_output.clone()
        return grad_output, None

class RoundWithGradient_ref(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bit_range, bit_range_f, T):
        x = x / bit_range
        x = x * bit_range_f
        x = x - 0.5

        coeff = torch.tanh(0.5 * T)
        out = x - torch.round(x)
        out = torch.tanh(T * out)
        out = out / coeff
        out = (out + 1) * 0.5

        out = out + torch.round(x)
        out = out / bit_range_f
        
        out = out * bit_range

        return out.round()
    @staticmethod
    def backward(ctx, g):
        return g, None, None, None

class RoundWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, g):
        return g

class FloorWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.floor()
    @staticmethod
    def backward(ctx, g):
        return g

class CeilWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.ceil()
    @staticmethod
    def backward(ctx, g):
        return g

class DSQConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                momentum = 0.1, num_bit = 3, QInput = True, bSetQ = True):
        super(DSQConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bit = num_bit
        self.quan_input = QInput
        self.bit_range = 2**self.num_bit -1
        self.is_quan = bSetQ
        self.temp = -1
        self.param_num = None
        self.feat_num = None
        self.pre_weight = None
        self.w_diff = None
        self.real_diff = None
        self.pre_bit_w = None

        if self.is_quan:
            # using int32 max/min as init and backprogation to optimization
            # Weight
            self.sW = nn.Parameter(data = torch.tensor(8).float())
            self.register_buffer('init', torch.tensor([1]).float())
            self.beta = nn.Parameter(data = torch.tensor(0.2).float())

            # Activation input
            if self.quan_input:
                self.uA = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
                # self.lA = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
                self.bit_w = nn.Parameter(data = torch.tensor(2).float())
                # self.bit_a = nn.Parameter(data = torch.tensor(8).float())

    # def quantizer(self, x, bit, bit_f):
    #     bit_range = (2 ** bit) - 1

    #     bit_floor = FloorWithGradient.apply(bit_f)
    #     bit_range_f = (2 ** bit_floor) - 1

    #     # out = RoundWithGradient.apply(x * bit_range) / bit_range
        
    #     T = torch.exp(8 * (bit_f.ceil() - bit_f)).detach()
        
    #     if bit_floor == 1:
    #         x = q_transform.apply(x * bit_range_f, T) / bit_range_f
    #     else:
    #         x = q_transform.apply(x * (bit_range_f - 1), T) / (bit_range_f - 1)

    #     if bit == 1:
    #         out = RoundWithGradient.apply(x * bit_range) / bit_range
    #     else:
    #         out = RoundWithGradient.apply(x * (bit_range - 1)) / (bit_range - 1)

    #     out = 2 * out - 1

    #     return out

    def quantizer(self, x, bit, bit_f):
        bit_range = (2 ** bit) - 1

        # bit_floor = bit_f.floor()
        bit_floor = FloorWithGradient.apply(bit_f)
        bit_range_f = (2 ** bit_floor) - 1

        if bit != 1:
            bit_range = bit_range - 1
        
        if bit_floor != 1:
            bit_range_f = bit_range_f - 1

        T = torch.exp(10 * (bit_f.ceil() - bit_f)).detach()

        # x = q_transform.apply(x * bit_range_f, T) / bit_range_f

        # out = RoundWithGradient_ref.apply(x * bit_range, bit_range.detach(), bit_range_f.detach(), T) / bit_range
        out = RoundWithGradient.apply(x * bit_range) / bit_range
        out = 2 * out - 1
        return out
        

    def w_quan(self, x, s):
        out = (x / torch.abs(s))
        out = (out + 1) * 0.5

        bit_w = RoundWithGradient.apply(self.bit_w)
        # bit_w = torch.tensor(2)
        # bit_range = torch.tensor((2 ** bit_w) - 1)
        
        # bit_f = FloorWithGradient.apply(self.bit_w)
        # bit_c = CeilWithGradient.apply(self.bit_w)

        out = torch.clamp(out, min=0, max=1)
        out_r = self.quantizer(out, bit_w, self.bit_w)

        diff = torch.mean(torch.abs(out_r - out_r))
        # diff = torch.mean(torch.abs(out_r - out_pre))

        return out_r, diff

    def a_quan(self, x, u, l):
        # bit_a = RoundWithGradient.apply(self.bit_a)
        # bit_range = (2 ** bit_a) - 1

        bit_a = torch.tensor(2)
        bit_range = torch.tensor((2 ** bit_a) - 1)

        delta = (u - l)
        interval = (x - l) / delta

        # output = SoftWithGradient.apply(interval, bit_range)
        output = torch.clamp(interval, min=0, max=1)
        output = RoundFunction.apply(output, bit_range)

        return output

    def cal_diff(self, pre_w, curr_w):
        diff = torch.abs(pre_w - curr_w)
        diff = diff.mean()
        return diff

    def forward(self, x):

        if self.is_quan:
            if self.init == 1:
                print(self.init)

                # self.bit_w.data = (self.weight.std() * 0) + bit_num
                # self.bit_a.data = (self.weight.std() * 0) + bit_num

                self.sW.data = self.weight.std() * 3
                self.uA.data = (x.std() / math.sqrt(1 - 2/math.pi)) * 3
                self.pre_weight = self.weight.detach()

            # Weight kernel_soft_argmax
            if self.bit_w <= 1:
                self.bit_w.data = 1.0 + (self.weight.std() * 0)

            # if self.bit_a <= 1:
            #     self.bit_a.data = 1.0 + (self.weight.std() * 0)

            if self.pre_weight == None:
                self.pre_weight = self.weight.detach()

            if self.pre_bit_w == None:
                self.pre_bit_w = self.bit_w.detach()

            Qweight, self.real_diff = self.w_quan(self.weight, self.sW)

            self.w_diff = self.cal_diff(self.pre_weight, Qweight)

            self.pre_weight = Qweight.detach()
            self.pre_bit_w = self.bit_w.round()

            Qbias = self.bias

            # Input(Activation)
            Qactivation = x
            if self.quan_input:
#                Qactivation = self.clipping(x, curr_running_ua, curr_running_la)
                Qactivation = self.a_quan(x, self.uA, 0)
                # Qactivation = torch.exp(self.beta_a) * Qactivation

            if self.init == 1:
                # print(self.init)
                # self.init = torch.tensor(0)
                q_output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
                ori_output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

                self.beta.data = torch.mean(torch.abs(ori_output)) / \
                                 torch.mean(torch.abs(q_output))
            
            if self.init == 0:
                self.param_num = (self.weight == self.weight).sum().detach()
                b, c, h, w = x.size()
                self.feat_num = torch.tensor(c * h * w).float().cuda().detach()

            output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
            output = torch.abs(self.beta) * output

        else:
            output =  F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

        return output

class DSQConv_a(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                momentum = 0.1, num_bit = 3, QInput = True, bSetQ = True):
        super(DSQConv_a, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bit = num_bit
        self.quan_input = QInput
        self.bit_range = 2**self.num_bit -1
        self.is_quan = bSetQ
        self.temp = -1
        self.param_num = None
        self.feat_num = None
        self.pre_weight = None
        self.w_diff = None
        self.real_diff = None
        self.pre_bit_w = None
        self.pre_bit_a = None
        self.epoch = 0
        self.total_epoch = 135150
        self.alpha = None
        self.iter = 0
        self.iter_w = 0
        self.iter_a = 0
        self.diff = 0
        self.diff_w = 0
        self.diff_a = 0

        if self.is_quan:
            # using int32 max/min as init and backprogation to optimization
            # Weight
            self.sW = nn.Parameter(data = torch.tensor(8).float())
            self.register_buffer('init', torch.tensor([1]).float())
            self.beta = nn.Parameter(data = torch.tensor(0.2).float())

            # Activation input
            if self.quan_input:
                self.uA = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
                # self.lA = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
                self.bit_w = nn.Parameter(data = torch.tensor(2.1).float())
                self.bit_a = nn.Parameter(data = torch.tensor(2.1).float())

    def quantizer(self, x, bit, bit_f):
        bit_range = (2 ** bit) - 1

        # bit_floor = bit_f.floor()

        if self.pre_bit_w.round() != self.bit_w.round():
            self.iter_w = 0
            bit_range_curr = (2 ** self.bit_w.round()) - 1
            bit_range_prev = (2 ** self.pre_bit_w.round()) - 1

            if self.pre_bit_w.round() != 1:
                bit_range_prev = bit_range_prev - 1

            if self.bit_w.round() != 1:
                bit_range_curr = bit_range_curr - 1

            Q_prev = (x * bit_range_prev).round() / bit_range_prev
            Q_curr = (x * bit_range_curr).round() / bit_range_curr

            self.diff_w = (Q_prev - Q_curr).abs().mean()
            scale = 1 - (self.diff_w * torch.exp(-self.iter_w / torch.tensor(1000.)))
            # self.diff_w = (Q_prev - Q_curr).abs()
            # scale = 1 + (self.diff_w * torch.exp(-self.iter_w / torch.tensor(1000.)))

        else:
            if self.training:
                self.iter_w = self.iter_w + 1
            scale = 1 - (self.diff_w * torch.exp(-self.iter_w / torch.tensor(1000.)))

        out = Quantizer_w.apply(x, bit)
        # out = Quantizer_w_scale.apply(x, bit, scale.detach())
        out = 2 * out - 1
        return out

    def quantizer_a(self, x, bit, bit_f):
        bit_range = (2 ** bit) - 1
        # bit_floor = bit_f.floor()

        if self.pre_bit_a.round() != self.bit_a.round():

            self.iter_a = 0
            bit_range_curr = (2 ** self.bit_a.round()) - 1
            bit_range_prev = (2 ** self.pre_bit_a.round()) - 1

            Q_prev = (x * bit_range_prev).round() / bit_range_prev
            Q_curr = (x * bit_range_curr).round() / bit_range_curr

            self.diff_a = (Q_prev - Q_curr).abs().mean()
            scale = 1 - self.diff_a * torch.exp(-self.iter_a / torch.tensor(1000.))
            # self.diff_a = (Q_prev - Q_curr).abs()
            # scale = 1 + self.diff_a * torch.exp(-self.iter_a / torch.tensor(1000.))

        else:
            if self.training:
                self.iter_a = self.iter_a + 1
            scale = 1 - self.diff_a * torch.exp(-self.iter_a / torch.tensor(1000.))

        out = Quantizer_a.apply(x, bit)
        # out = Quantizer_a_scale.apply(x, bit, scale.detach())
        return out

    def w_quan(self, x, s):
        out = (x / torch.abs(s))
        out = (out + 1) * 0.5

        bit_w = RoundWithGradient.apply(self.bit_w)

        out = torch.clamp(out, min=0, max=1)
        out_r = self.quantizer(out, bit_w, self.bit_w)

        diff = torch.mean(torch.abs(out_r - out_r))

        return out_r, diff

    def a_quan(self, x, u, l):
        bit_a = RoundWithGradient.apply(self.bit_a)
        bit_range = (2 ** bit_a) - 1

        # bit_a = torch.tensor(2)
        # bit_range = torch.tensor((2 ** bit_a) - 1)

        delta = (u - l)
        interval = (x - l) / delta

        # output = SoftWithGradient.apply(interval, bit_range)
        out = torch.clamp(interval, min=0, max=1)
        output = self.quantizer_a(out, bit_a, self.bit_a)
        
        return output

    def cal_diff(self, pre_w, curr_w):
        diff = torch.abs(pre_w - curr_w)
        diff = diff.mean()
        return diff

    def forward(self, x):

        if self.is_quan:
            if self.init == 1:
                print(self.init)
                # self.init = torch.tensor(0)

                # self.bit_w.data = (self.weight.std() * 0) + bit_num
                # self.bit_a.data = (self.weight.std() * 0) + bit_num

                self.sW.data = self.weight.std() * 3
                self.uA.data = (x.std() / math.sqrt(1 - 2/math.pi)) * 3
                self.pre_weight = self.weight.detach()

            if self.pre_bit_w == None:
                self.pre_bit_w = self.bit_w.clone()
            
            if self.pre_bit_a == None:
                self.pre_bit_a = self.bit_a.clone()

            # self.alpha = 1 - (self.epoch / self.total_epoch)
            mu = 10
            self.alpha = 10 + (mu * torch.log10(torch.log10(10 - 9*(torch.tensor(self.epoch) / self.total_epoch)) + 1e-6))

            # Weight kernel_soft_argmax
            if self.bit_w <= 1:
                self.bit_w.data = 1.0 + (self.weight.std() * 0)

            if self.bit_a <= 1:
                self.bit_a.data = 1.0 + (self.weight.std() * 0)

            if self.pre_weight == None:
                self.pre_weight = self.weight.detach()

            Qweight, self.real_diff = self.w_quan(self.weight, self.sW)

            self.w_diff = self.cal_diff(self.pre_weight, Qweight)

            self.pre_weight = Qweight.detach()

            Qbias = self.bias

            # Input(Activation)
            Qactivation = x
            if self.quan_input:
#                Qactivation = self.clipping(x, curr_running_ua, curr_running_la)
                Qactivation = self.a_quan(x, self.uA, 0)
                # Qactivation = torch.exp(self.beta_a) * Qactivation

            if self.init == 1:
                # print(self.init)
                # self.init = torch.tensor(0)
                q_output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
                ori_output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

                self.beta.data = torch.mean(torch.abs(ori_output)) / \
                                 torch.mean(torch.abs(q_output))
            
            if self.init == 0:
                b, c, h, w = x.size()
                self.feat_num = torch.tensor(c * h * w).float().cuda().detach()
                self.param_num = (self.weight == self.weight).sum().detach()

            self.pre_bit_w = self.bit_w.clone()
            self.pre_bit_a = self.bit_a.clone()

            output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
            output = torch.abs(self.beta) * output

        else:
            output =  F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

        return output

class DSQConv_8bit(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                momentum = 0.1, num_bit = 3, QInput = True, bSetQ = True):
        super(DSQConv_8bit, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bit = num_bit
        self.quan_input = QInput
        self.bit_range = 2**self.num_bit -1
        self.is_quan = bSetQ
        self.temp = -1
        self.param_num = None
        self.feat_num = None
        self.bit_w = torch.tensor(8).float()
        self.bit_a = torch.tensor(8).float()

        if self.is_quan:
            # using int32 max/min as init and backprogation to optimization
            # Weight
            self.sW = nn.Parameter(data = torch.tensor(8).float())
            self.register_buffer('init', torch.tensor([1]).float())
            self.beta = nn.Parameter(data = torch.tensor(0.2).float())

            # Activation input
            if self.quan_input:
                self.uA = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
                self.lA = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
                # self.bit_w = nn.Parameter(data = torch.tensor(8).float())
                # self.bit_a = nn.Parameter(data = torch.tensor(8).float())

    def w_quan(self, x, s):
        out = (x / torch.abs(s))
        out = (out + 1) * 0.5

        # bit_w = RoundWithGradient.apply(self.bit_w)
        # bit_range = (2 ** bit_w) - 1

        bit_w = torch.tensor(8)
        bit_range = torch.tensor((2 ** bit_w) - 1)

        if bit_w == 1:
            out = torch.clamp(out, min=0, max=1)
            out = RoundFunction.apply(out, bit_range)
        
        else:
            out = torch.clamp(out, min=0, max=1)
            out = RoundFunction.apply(out, bit_range - 1)
        
        out = 2 * out - 1

        return out

    def a_quan(self, x, u, l):
        # bit_a = RoundWithGradient.apply(self.bit_a)
        # bit_range = (2 ** bit_a) - 1

        bit_a = torch.tensor(8)
        bit_range = torch.tensor((2 ** bit_a) - 1)

        delta = (u - l)
        interval = (x - l) / delta

        # output = SoftWithGradient.apply(interval, bit_range)
        output = torch.clamp(interval, min=0, max=1)
        output = RoundFunction.apply(output, bit_range)
        output = 2 * output - 1

        return output

    def forward(self, x):

        if self.is_quan:
            if self.init == 1:
                print(self.init)

                # self.bit_w.data = (self.weight.std() * 0) + bit_num
                # self.bit_a.data = (self.weight.std() * 0) + bit_num

                self.sW.data = self.weight.std() * 3
                self.uA.data = (x.std() / math.sqrt(1 - 2/math.pi)) * 3
                self.lA.data = -(x.std() / math.sqrt(1 - 2/math.pi)) * 3

            # Weight kernel_soft_argmax
            # if self.bit_w <= 1:
            #     self.bit_w.data = 1.0 + (self.weight.std() * 0)

            # if self.bit_a <= 1:
            #     self.bit_a.data = 1.0 + (self.weight.std() * 0)

            Qweight = self.w_quan(self.weight, self.sW)
            Qbias = self.bias

            # Input(Activation)
            Qactivation = x
            if self.quan_input:
#                Qactivation = self.clipping(x, curr_running_ua, curr_running_la)
                Qactivation = self.a_quan(x, self.uA, self.lA)
                # Qactivation = torch.exp(self.beta_a) * Qactivation

            if self.init == 1:
                # print(self.init)
                # self.init = torch.tensor(0)
                q_output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
                ori_output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

                self.beta.data = torch.mean(torch.abs(ori_output)) / \
                                 torch.mean(torch.abs(q_output))
            
            if self.init == 0:
            #     self.param_num = (self.weight == self.weight).sum().detach()
                b, c, h, w = x.size()
                self.param_num = (self.weight == self.weight).sum().detach()
                self.feat_num = torch.tensor(c * h * w).float().cuda().detach()

            output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
            output = torch.abs(self.beta) * output

        else:
            output =  F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

        return output



class DSQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                num_bit = 4, bSetQ = True, QInput=True):
        super(DSQLinear, self).__init__(in_features, out_features, bias)
        self.num_bit = num_bit
        self.quan_input = QInput
        self.bit_range = 2**self.num_bit -1
        self.is_quan = bSetQ
        self.temp = -1
        self.bias_T = bias
        self.bit_w = torch.tensor(8).float()
        self.bit_a = torch.tensor(8).float()

        if self.is_quan:
            # using int32 max/min as init and backprogation to optimization
            # Weight
            self.sW = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
            self.register_buffer('init', torch.tensor(1).float())
            self.beta = nn.Parameter(data = torch.tensor(0.2).float())

            # Activation input
            if self.quan_input:
                self.uA = nn.Parameter(data = torch.tensor(2 ** 31 - 1).float())
                # self.lA = nn.Parameter(data = torch.tensor(0).float())
            
            if self.bias_T:
                self.sB = nn.Parameter(data = torch.tensor(2 ** 31 - 1).float())
                self.betaB = nn.Parameter(data = torch.tensor(0.2).float())

    def w_quan(self, x, s):
        out = (x / torch.abs(s))
        out = (out + 1) * 0.5

        # bit_w = RoundWithGradient.apply(self.bit_w)
        # bit_range = (2 ** bit_w) - 1

        bit_w = torch.tensor(8)
        bit_range = torch.tensor((2 ** bit_w) - 1)

        if bit_w == 1:
            out = torch.clamp(out, min=0, max=1)
            out = RoundFunction.apply(out, bit_range)
        
        else:
            out = torch.clamp(out, min=0, max=1)
            out = RoundFunction.apply(out, bit_range - 1)
        
        out = 2 * out - 1

        return out

    def a_quan(self, x, u, l):
        # bit_a = RoundWithGradient.apply(self.bit_a)
        # bit_range = (2 ** bit_a) - 1

        bit_a = torch.tensor(8)
        bit_range = torch.tensor((2 ** bit_a) - 1)

        delta = (u - l)
        interval = (x - l) / delta

        # output = SoftWithGradient.apply(interval, bit_range)
        output = torch.clamp(interval, min=0, max=1)
        output = RoundFunction.apply(output, bit_range)

        return output

    def forward(self, x):

        if self.is_quan:
            if self.init == 1:
                print(self.init)
                # self.init = torch.tensor(0)
                self.sW.data = self.weight.std() * 3
                self.uA.data = (x.std() / math.sqrt(1 - 2/math.pi)) * 3

                if self.bias_T:
                    self.sB.data = self.bias.std() * 3

            if self.quan_input:
                curr_running_la = 0
                curr_running_ua = self.uA

            Qweight = self.w_quan(self.weight, self.sW)

            if self.bias_T:
                Qbias = self.w_quan(self.bias, self.sB) * self.betaB
            else:
                Qbias = self.bias

            # Input(Activation)
            Qactivation = x
            
            if self.quan_input:
                Qactivation = self.a_quan(Qactivation, curr_running_ua, curr_running_la)

            if self.init == 1:
                q_output = F.linear(Qactivation, Qweight, Qbias)
                ori_output = F.linear(x, self.weight, self.bias)

                self.beta.data = torch.mean(torch.abs(ori_output)) / \
                                 torch.mean(torch.abs(q_output))

                self.betaB.data = torch.mean(torch.abs(self.bias)) / \
                                  torch.mean(torch.abs(Qbias))

            if self.init == 0:
                b, cin = x.size()
                self.feat_num = torch.tensor(cin).float().cuda().detach()
                self.param_num = (self.weight == self.weight).sum().detach() + (self.bias == self.bias).sum().detach()

            output = F.linear(Qactivation, Qweight, Qbias)
            output = torch.abs(self.beta) * output

        else:
            output =  F.linear(x, self.weight, self.bias)

        return output