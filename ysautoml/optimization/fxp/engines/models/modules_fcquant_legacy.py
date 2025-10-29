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

        return g, grad_b * g

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

        return g, grad_b * g

class RoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n):
        return torch.round(x * n) / n
    @staticmethod
    def backward(ctx, grad_output):
        # grad_input = grad_output.clone()
        return grad_output, None

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

            self.pre_bit_w = self.bit_w.clone()
            self.pre_bit_a = self.bit_a.clone()

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
        self.norm_weight = None
        self.w_diff = None
        self.real_diff = None
        self.pre_bit_w = None
        self.pre_bit_a = None
        self.pre_bit_w_fix = None
        self.pre_bit_a_fix = None
        self.output_mean = None
        self.output_std = None
        self.buff_weight = None
        self.buff_act = None

        self.bit_w_grad = None
        self.bit_a_grad = None

        self.update_w = 1
        self.update_a = 1

        self.bit_w_grad_real = torch.tensor(0.)
        self.bit_a_grad_real = torch.tensor(0.)

        self.bit_w_grad_list = torch.zeros(9)
        self.bit_a_grad_list = torch.zeros(9)

        self.bit_w_moment = torch.tensor(0.)
        self.bit_a_moment = torch.tensor(0.)

        self.bit_w_direction = torch.tensor(0.)
        self.bit_a_direction = torch.tensor(0.)

        self.bit_w_score = torch.tensor(0.)
        self.bit_a_score = torch.tensor(0.)
        
        self.alpha_w_gauge = torch.tensor(0.)
        self.alpha_a_gauge = torch.tensor(0.)

        # self.param_num_total = torch.tensor(0.)
        # self.param_num_pow_total = torch.tensor(0.)

        self.register_buffer('param_num_total', torch.tensor(0.).float())
        self.register_buffer('param_num_pow_total', torch.tensor(0.).float())

        self.register_buffer('feat_num_total', torch.tensor(0.).float())
        self.register_buffer('feat_num_pow_total', torch.tensor(0.).float())

        self.bit_w_diff = torch.tensor(0.)
        self.bit_a_diff = torch.tensor(0.)

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
                self.bit_w = nn.Parameter(data = torch.tensor(2.0).float())
                self.bit_a = nn.Parameter(data = torch.tensor(2.0).float())
    def round_bit_w(self, x, bit):
        if bit < 1:
            bit = 1

        bit_range = (2 ** bit) - 1

        if bit != 1:
        	bit_range = bit_range - 1

        out = (x * bit_range).round() / bit_range

        return out

    def round_bit_a(self, x, bit):
        if bit < 1:
            bit = 1

        bit_range = (2 ** bit) - 1
        out = (x * bit_range).round() / bit_range

        return out
    def quantizer(self, x, bit):
        bit_range = (2 ** bit) - 1
        if bit != 1:
            bit_range = bit_range - 1

        if self.training:
            if self.bit_w.round() != self.pre_bit_w.round():
                self.bit_w_score = 0.99 * self.bit_w_score + 0.01

            else:
                self.bit_w_score = 0.99 * self.bit_w_score

        out = Quantizer_w.apply(x, bit)
        diff = x - out
        bit_w_grad = (diff / (bit_range**2)) * (2**bit) * torch.log(torch.tensor(2.0))
        self.bit_w_grad = bit_w_grad.abs().mean()
        out = 2 * out - 1
        return out

    def quantizer_a(self, x, bit):
        bit_range = (2 ** bit) - 1

        if self.training:
            if self.bit_a.round() != self.pre_bit_a.round():
                self.bit_a_score = (0.99 * self.bit_a_score) + 0.01

            else:
                self.bit_a_score = (0.99 * self.bit_a_score)

        out = Quantizer_a.apply(x, bit)
        diff = x - out
        bit_a_grad = (diff / (bit_range**2)) * (2**bit) * torch.log(torch.tensor(2.0))
        self.bit_a_grad = bit_a_grad.abs().mean()
        return out

    def w_quan(self, x, s):
        out = (x / torch.abs(s))
        out = (out + 1) * 0.5

        self.norm_weight = out.detach()

        bit_w = RoundWithGradient.apply(self.bit_w)

        out = torch.clamp(out, min=0, max=1)
        out_r = self.quantizer(out, bit_w)

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
        output = self.quantizer_a(out, bit_a)

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
                self.norm_weight = self.weight.detach()

            # if self.bit_w_diff > 0:
            #     lambda_w = (self.param_num_total / self.param_num_pow_total) * self.bit_w_diff
            #     self.bit_w.data = self.bit_w - (lambda_w.detach() * (self.param_num / self.param_num_total))

            # if self.bit_a_diff > 0:
            #     lambda_a = (self.feat_num_total / self.feat_num_pow_total) * self.bit_a_diff
            #     self.bit_a.data = self.bit_a - (lambda_a.detach() * (self.feat_num / self.feat_num_total))

            if self.pre_bit_w == None:
                self.pre_bit_w = self.bit_w.clone()
                self.bit_w_moment = self.bit_w.clone()

            if self.pre_bit_w_fix == None:
                self.pre_bit_w_fix = self.bit_w.round()

            if self.pre_bit_a == None:
                self.pre_bit_a = self.bit_a.clone()
                self.bit_a_moment = self.bit_a.clone()

            if self.pre_bit_a_fix == None:
                self.pre_bit_a_fix = self.bit_a.round()

            # alpha = (self.epoch ** (0.1))
            lambda_alpha = 2.0
            pow_alpha = 1.
            alpha = lambda_alpha * (self.epoch ** (pow_alpha))

            if alpha < 1:
                alpha = 0

            elif alpha > 1:
                alpha = 1

            # if self.update_w:
            #     self.update_w = 0
            #     self.bit_w_grad_list[self.pre_bit_w.round().int().item()] = (0.99) * self.bit_w_grad_list[self.pre_bit_w.round().int().item()] + \
            #                                                                 (0.01) * self.bit_w_grad_real

            #     if self.bit_w.round() != self.pre_bit_w.round():
            #         direction = torch.sign(self.bit_w_grad_list[self.pre_bit_w.round().int().item()]) * \
            #                     torch.sign(self.bit_w_grad_list[self.bit_w.round().int().item()])

            #         w_diff_grad = self.bit_w_grad_list[self.pre_bit_w.round().int().item()].abs() < self.bit_w_grad_list[self.bit_w.round().int().item()].abs()

            #         if direction < 0 and w_diff_grad:
            #             self.bit_w.data = self.bit_w - alpha * (self.bit_w - self.pre_bit_w)
            
            # if self.update_a:
            #     self.update_a = 0
            #     self.bit_a_grad_list[self.pre_bit_a.round().int().item()] = (0.99) * self.bit_a_grad_list[self.pre_bit_a.round().int().item()] + \
            #                                                                 (0.01) * self.bit_a_grad_real

            #     if self.bit_a.round() != self.pre_bit_a.round():
            #         direction = torch.sign(self.bit_a_grad_list[self.pre_bit_a.round().int().item()]) * \
            #                     torch.sign(self.bit_a_grad_list[self.bit_a.round().int().item()])

            #         a_diff_grad = self.bit_a_grad_list[self.pre_bit_a.round().int().item()].abs() < self.bit_a_grad_list[self.bit_a.round().int().item()].abs()

            #         if direction < 0 and a_diff_grad:
            #             self.bit_a.data = self.bit_a - alpha * (self.bit_a - self.pre_bit_a)
            
            bit_w_curr_round = self.bit_w.round().clone()
            if self.update_w:
                self.update_w = 0
                if self.bit_w.round() != self.pre_bit_w.round():
                    direction = torch.sign(self.bit_w_grad_list[self.pre_bit_w.round().int().item()]) * \
                                torch.sign(self.bit_w_grad_list[self.bit_w.round().int().item()])

                    w_diff_grad = self.bit_w_grad_list[self.pre_bit_w.round().int().item()].abs() < self.bit_w_grad_list[self.bit_w.round().int().item()].abs()

                    if direction < 0 and w_diff_grad:
                        self.bit_w.data = self.bit_w - alpha * (self.bit_w - self.pre_bit_w)

                if alpha == 0:
                    self.bit_w_grad_list[self.pre_bit_w.round().int().item()] = (0.99) * self.bit_w_grad_list[self.pre_bit_w.round().int().item()] + \
                                                                                (0.01) * self.bit_w_grad_real
            bit_a_curr_round = self.bit_a.round().clone()
            if self.update_a:
                self.update_a = 0

                if self.bit_a.round() != self.pre_bit_a.round():
                    direction = torch.sign(self.bit_a_grad_list[self.pre_bit_a.round().int().item()]) * \
                                torch.sign(self.bit_a_grad_list[self.bit_a.round().int().item()])

                    a_diff_grad = self.bit_a_grad_list[self.pre_bit_a.round().int().item()].abs() < self.bit_a_grad_list[self.bit_a.round().int().item()].abs()

                    if direction < 0 and a_diff_grad:
                        self.bit_a.data = self.bit_a - alpha * (self.bit_a - self.pre_bit_a)
            
                if alpha == 0:
                    self.bit_a_grad_list[self.pre_bit_a.round().int().item()] = (0.99) * self.bit_a_grad_list[self.pre_bit_a.round().int().item()] + \
                                                                                (0.01) * self.bit_a_grad_real

            if self.training:
                self.update_w = 1
                self.update_a = 1

            self.bit_w_moment = ((self.bit_w_moment * 0.99) + (self.bit_w * 0.01)).detach()
            self.bit_w_direction = torch.sign(self.bit_w_moment - self.bit_w).detach()

            self.bit_a_moment = ((self.bit_a_moment * 0.99) + (self.bit_a * 0.01)).detach()
            self.bit_a_direction = torch.sign(self.bit_a_moment - self.bit_a).detach()

            # self.alpha = 1 - (self.epoch / self.total_epoch)
            mu = 10
            self.alpha = 10 + (mu * torch.log10(torch.log10(10 - 9*(torch.tensor(self.epoch) / self.total_epoch)) + 1e-6))

            # Weight kernel_soft_argmax
            if self.bit_w <= 1:
                self.bit_w.data = 1.0 + (self.weight.std() * 0)

            if self.bit_a <= 1:
                self.bit_a.data = 1.0 + (self.weight.std() * 0)

            if self.bit_w >= 8:
                self.bit_w.data = 8.0 + (self.weight.std() * 0)

            if self.bit_a >= 8:
                self.bit_a.data = 8.0 + (self.weight.std() * 0)

            if self.pre_weight == None:
                self.pre_weight = self.weight.detach()

            ## directions of latent bit-width
            # self.bit_w_grad = self.bit_w - self.pre_bit_w
            # self.bit_a_grad = self.bit_a - self.pre_bit_a

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

                q_output_mean = q_output.mean(dim=[0,2,3], keepdim=True)
                q_output_std = q_output.std(dim=[0,2,3], keepdim=True)

                # q_output = (q_output - q_output_mean) / (q_output_std + 1e-3)

                ori_output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

                self.beta.data = torch.mean(torch.abs(ori_output)) / \
                                 torch.mean(torch.abs(q_output))

            if self.init == 1 or self.init == 0:
                b, c, h, w = x.size()
                self.feat_num = torch.tensor(c * h * w).float().cuda().detach()
                self.param_num = (self.weight == self.weight).sum().detach()

            self.pre_bit_w = self.bit_w.clone()
            self.pre_bit_a = self.bit_a.clone()

            output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)

            output_mean = output.mean(dim=[0,2,3], keepdim=True)
            output_std = output.std(dim=[0,2,3], keepdim=True)

            self.output_mean = output_mean.abs().mean()
            self.output_std = output_std.abs().mean()

            # output = (output - output_mean) / (output_std + 1e-3)

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
        self.bit_w_score = torch.tensor(0.)
        self.bit_a_score = torch.tensor(0.)

        self.bit_w_moment = torch.tensor(0.)
        self.bit_a_moment = torch.tensor(0.)

        self.bit_w_direction = torch.tensor(0.)
        self.bit_a_direction = torch.tensor(0.)

        self.register_buffer('param_num_total', torch.tensor(0.).float())
        self.register_buffer('param_num_pow_total', torch.tensor(0.).float())

        self.register_buffer('feat_num_total', torch.tensor(0.).float())
        self.register_buffer('feat_num_pow_total', torch.tensor(0.).float())


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

            if self.init == 1 or self.init == 0:
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
        # self.bit_w = torch.tensor(8).float()
        # self.bit_a = torch.tensor(8).float()
        self.bit_w_score = torch.tensor(0.)
        self.bit_a_score = torch.tensor(0.)

        self.param_num = None
        self.feat_num = None

        self.bit_w_grad = None
        self.bit_a_grad = None

        self.pre_bit_w = None
        self.pre_bit_a = None

        self.update_w = 1
        self.update_a = 1

        self.alpha_w_gauge = torch.tensor(0.)
        self.alpha_a_gauge = torch.tensor(0.)

        self.bit_w_grad_real = torch.tensor(0.)
        self.bit_a_grad_real = torch.tensor(0.)

        self.bit_w_grad_list = torch.zeros(9)
        self.bit_a_grad_list = torch.zeros(9)

        self.bit_w_moment = torch.tensor(0.)
        self.bit_a_moment = torch.tensor(0.)

        self.bit_w_direction = torch.tensor(0.)
        self.bit_a_direction = torch.tensor(0.)

        self.register_buffer('param_num_total', torch.tensor(0.).float())
        self.register_buffer('param_num_pow_total', torch.tensor(0.).float())

        self.register_buffer('feat_num_total', torch.tensor(0.).float())
        self.register_buffer('feat_num_pow_total', torch.tensor(0.).float())

        self.epoch = 0


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
                self.bit_w = nn.Parameter(data = torch.tensor(2.0).float())
                self.bit_a = nn.Parameter(data = torch.tensor(2.0).float())

    def w_quan(self, x, s):
        out = (x / torch.abs(s))
        out = (out + 1) * 0.5

        bit_w = RoundWithGradient.apply(self.bit_w)
        bit_range = (2 ** bit_w) - 1

        # bit_w = torch.tensor(8)
        # bit_range = torch.tensor((2 ** bit_w) - 1)

        if bit_w == 1:
            out = torch.clamp(out, min=0, max=1)
            out = RoundWithGradient.apply(out * bit_range) / bit_range
            # out = RoundFunction.apply(out, bit_range)

        else:
            out = torch.clamp(out, min=0, max=1)
            out = RoundWithGradient.apply(out * (bit_range-1)) / (bit_range-1)
            # out = RoundFunction.apply(out, bit_range - 1)

        out = 2 * out - 1

        return out

    def a_quan(self, x, u, l):
        bit_a = RoundWithGradient.apply(self.bit_a)
        bit_range = (2 ** bit_a) - 1

        # bit_a = torch.tensor(8)
        # bit_range = torch.tensor((2 ** bit_a) - 1)

        delta = (u - l)
        interval = (x - l) / delta

        # output = SoftWithGradient.apply(interval, bit_range)
        output = torch.clamp(interval, min=0, max=1)
        # output = RoundFunction.apply(output, bit_range)
        output = RoundWithGradient.apply(output * bit_range) / bit_range

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

            if self.pre_bit_w == None:
                self.pre_bit_w = self.bit_w.clone()

            if self.pre_bit_a == None:
                self.pre_bit_a = self.bit_a.clone()
            ###############
            lambda_alpha = 2.0
            pow_alpha = 1.
            alpha = lambda_alpha * (self.epoch ** (pow_alpha))
            
            if alpha < 1:
                alpha = 0

            elif alpha > 1:
                alpha = 1



            # if self.update_w:
            #     self.update_w = 0
            #     self.bit_w_grad_list[self.pre_bit_w.round().int().item()] = (0.99) * self.bit_w_grad_list[self.pre_bit_w.round().int().item()] + \
            #                                                                 (0.01) * self.bit_w_grad_real

            #     if self.bit_w.round() != self.pre_bit_w.round():
            #         direction = torch.sign(self.bit_w_grad_list[self.pre_bit_w.round().int().item()]) * \
            #                     torch.sign(self.bit_w_grad_list[self.bit_w.round().int().item()])

            #         w_diff_grad = self.bit_w_grad_list[self.pre_bit_w.round().int().item()].abs() < self.bit_w_grad_list[self.bit_w.round().int().item()].abs()

            #         if direction < 0 and w_diff_grad:
            #             self.bit_w.data = self.bit_w - alpha * (self.bit_w - self.pre_bit_w)
            
            # if self.update_a:
            #     self.update_a = 0
            #     self.bit_a_grad_list[self.pre_bit_a.round().int().item()] = (0.99) * self.bit_a_grad_list[self.pre_bit_a.round().int().item()] + \
            #                                                                 (0.01) * self.bit_a_grad_real

            #     if self.bit_a.round() != self.pre_bit_a.round():
            #         direction = torch.sign(self.bit_a_grad_list[self.pre_bit_a.round().int().item()]) * \
            #                     torch.sign(self.bit_a_grad_list[self.bit_a.round().int().item()])

            #         a_diff_grad = self.bit_a_grad_list[self.pre_bit_a.round().int().item()].abs() < self.bit_a_grad_list[self.bit_a.round().int().item()].abs()

            #         if direction < 0 and a_diff_grad:
            #             self.bit_a.data = self.bit_a - alpha * (self.bit_a - self.pre_bit_a)

            bit_w_curr_round = self.bit_w.round().clone()
            if self.update_w:
                self.update_w = 0
                if self.bit_w.round() != self.pre_bit_w.round():
                    direction = torch.sign(self.bit_w_grad_list[self.pre_bit_w.round().int().item()]) * \
                                torch.sign(self.bit_w_grad_list[self.bit_w.round().int().item()])

                    w_diff_grad = self.bit_w_grad_list[self.pre_bit_w.round().int().item()].abs() < self.bit_w_grad_list[self.bit_w.round().int().item()].abs()

                    if direction < 0 and w_diff_grad:
                        self.bit_w.data = self.bit_w - alpha * (self.bit_w - self.pre_bit_w)

                if bit_w_curr_round == self.bit_w.round():
                    self.bit_w_grad_list[self.pre_bit_w.round().int().item()] = (0.99) * self.bit_w_grad_list[self.pre_bit_w.round().int().item()] + \
                                                                                (0.01) * self.bit_w_grad_real
            
            bit_a_curr_round = self.bit_a.round().clone()

            if self.update_a:
                self.update_a = 0

                if self.bit_a.round() != self.pre_bit_a.round():
                    direction = torch.sign(self.bit_a_grad_list[self.pre_bit_a.round().int().item()]) * \
                                torch.sign(self.bit_a_grad_list[self.bit_a.round().int().item()])

                    a_diff_grad = self.bit_a_grad_list[self.pre_bit_a.round().int().item()].abs() < self.bit_a_grad_list[self.bit_a.round().int().item()].abs()

                    if direction < 0 and a_diff_grad:
                        self.bit_a.data = self.bit_a - alpha * (self.bit_a - self.pre_bit_a)
            
                if self.bit_a.round() == bit_a_curr_round:
                    self.bit_a_grad_list[self.pre_bit_a.round().int().item()] = (0.99) * self.bit_a_grad_list[self.pre_bit_a.round().int().item()] + \
                                                                                (0.01) * self.bit_a_grad_real

            if self.training:
                self.update_w = 1
                self.update_a = 1

            if self.bit_w <= 1:
                self.bit_w.data = 1.0 + (self.weight.std() * 0)

            if self.bit_a <= 1:
                self.bit_a.data = 1.0 + (self.weight.std() * 0)

            if self.bit_w >= 8:
                self.bit_w.data = 8.0 + (self.weight.std() * 0)

            if self.bit_a >= 8:
                self.bit_a.data = 8.0 + (self.weight.std() * 0)

            ###############

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

            if self.init == 1 or self.init == 0:
                b, cin = x.size()
                self.feat_num = torch.tensor(cin).float().cuda().detach()
                self.param_num = (self.weight == self.weight).sum().detach() + (self.bias == self.bias).sum().detach()

            self.pre_bit_w = self.bit_w.clone().detach()
            self.pre_bit_a = self.bit_a.clone().detach()

            output = F.linear(Qactivation, Qweight, Qbias)
            output = torch.abs(self.beta) * output

        else:
            output =  F.linear(x, self.weight, self.bias)

        return output
