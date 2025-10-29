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

class RoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n):
        return torch.round(x * n) / n
    @staticmethod
    def backward(ctx, grad_output):
        # grad_input = grad_output.clone()
        return grad_output, None

class EWGS_discretizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in, n):
        x = x_in * (n)
        x = torch.round(x)
        x_out = x / (n)
        
        ctx.save_for_backward(x_in-x_out)
        return x_out
    @staticmethod
    def backward(ctx, g):
        diff = ctx.saved_tensors[0]
        scale = 1 + (1e-3) * torch.sign(g)*diff
        return g * scale, None, None

# class DSQConv_a(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
#                 momentum = 0.1, num_bit = 3, QInput = True, bSetQ = True):
#         super(DSQConv_a, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
#         self.num_bit = num_bit
#         self.quan_input = QInput
#         self.bit_range = 2**self.num_bit -1
#         self.is_quan = bSetQ
#         self.temp = -1

#         self.w_mask_0 = None
#         self.w_mask_1 = None
#         self.w_mask_2 = None
#         self.w_mask_3 = None
#         self.w_mask_4 = None

#         self.w_num_0 = None
#         self.w_num_1 = None
#         self.w_num_2 = None
#         self.w_num_3 = None

#         self.a_mask_1 = None
#         self.a_mask_2 = None
#         self.a_mask_3 = None

#         self.a_num_0 = None
#         self.a_num_1 = None
#         self.a_num_2 = None
#         self.a_num_3 = None

#         self.w_grad_0 = torch.tensor(0.0)
#         self.w_grad_1 = torch.tensor(0.0)
#         self.w_grad_2 = torch.tensor(0.0)
#         self.w_grad_3 = torch.tensor(0.0)
#         self.w_grad_4 = torch.tensor(0.0)

#         self.a_grad_1 = torch.tensor(0.0)
#         self.a_grad_2 = torch.tensor(0.0)
#         self.a_grad_3 = torch.tensor(0.0)

#         self.weight_norm = None
#         self.act_norm = None

#         if self.is_quan:
#             # using int32 max/min as init and backprogation to optimization
#             # Weight
#             self.sW = nn.Parameter(data = torch.tensor(8).float())
#             self.register_buffer('init', torch.tensor([1]).float())
#             self.beta = nn.Parameter(data = torch.tensor(0.2).float())

#             # Activation input
#             if self.quan_input:
#                 self.uA = nn.Parameter(data = torch.tensor(2 **31 - 1).float())

#     def w_quan(self, x, s):
#         out = (x / torch.abs(s))
#         out = (out + 1) * 0.5

#         bit = 2
#         bit_range = (2 ** bit) - 1
        
#         out_c = torch.clamp(out, min=0, max=1)
#         self.weight_norm = out_c
#         if self.training:
#             self.weight_norm.retain_grad()
#         out = RoundFunction.apply(self.weight_norm, bit_range)

#         out_compare = (self.weight_norm * bit_range).floor() / bit_range
        
#         self.w_mask_1 = ((out_compare == (0/bit_range)).float() - (self.weight_norm == 0).float()).bool()
#         self.w_mask_2 = (out_compare == (1/bit_range))
#         self.w_mask_3 = ((out_compare == (2/bit_range)).float() - (self.weight_norm == 1).float()).bool()
        

#         self.w_num_0 = (out == (0/bit_range))
#         self.w_num_1 = (out == (1/bit_range))
#         self.w_num_2 = (out == (2/bit_range))
#         self.w_num_3 = (out == (3/bit_range))

#         out = (2 * out) - 1

#         return out

#     def a_quan(self, x, u, l):
#         delta = (u - l)
#         out = (x - l) / delta

#         bit = 2
#         bit_range = (2 ** bit) - 1

#         self.act_norm = torch.clamp(out, min=0, max=1)
#         if self.training:
#             self.act_norm.retain_grad()
#         out = RoundFunction.apply(self.act_norm, bit_range)

#         out_compare = (self.act_norm * bit_range).floor() / bit_range

#         self.a_mask_1 = ((out_compare == (0/bit_range)).float() - (self.act_norm == 0).float()).bool()
#         self.a_mask_2 = (out_compare == (1/bit_range))
#         self.a_mask_3 = ((out_compare == (2/bit_range)).float() - (self.act_norm == 1).float()).bool()

#         self.a_num_0 = (out == (0/bit_range))
#         self.a_num_1 = (out == (1/bit_range))
#         self.a_num_2 = (out == (2/bit_range))
#         self.a_num_3 = (out == (3/bit_range))

#         return out

#     def forward(self, x):

#         if self.is_quan:
#             if self.init == 1:
#                 print(self.init)
#                 self.sW.data = self.weight.std() * 3
#                 self.uA.data = (x.std() / math.sqrt(1 - 2/math.pi)) * 3

#             Qweight = self.w_quan(self.weight, self.sW)
#             Qbias = self.bias

#             # Input(Activation)
#             Qactivation = x

#             if self.quan_input:
#                 Qactivation = self.a_quan(x, self.uA, 0)

#             if self.init == 1:
#                 # print(self.init)
#                 self.init = torch.tensor(0)
#                 q_output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
#                 ori_output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

#                 self.beta.data = torch.mean(torch.abs(ori_output)) / \
#                                  torch.mean(torch.abs(q_output))

#             output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
#             output = torch.abs(self.beta) * output

#         else:
#             output =  F.conv2d(x, self.weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)

#         return output

class Distribution(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, distribute_grad):
        ctx.save_for_backward(distribute_grad)
        return x
    @staticmethod
    def backward(ctx, grad_output):
        distribute_grad, = ctx.saved_tensors
        grad_mean = grad_output.abs().mean()
        scale = 5000
        grad_out = grad_output + (scale * distribute_grad * grad_mean)
        return grad_out, None

class Decay_aware(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s):
        ctx.save_for_backward(x, s)
        return x
    @staticmethod
    def backward(ctx, grad_output):
        x, s, = ctx.saved_tensors
        mask = (x <= -s).float() + (x >= s).float()
        decay_sup = mask * (-x * 1e-4)
        return grad_output + decay_sup, None

class Interval_aware(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g):
        ctx.save_for_backward(x, g)
        return x
    @staticmethod
    def backward(ctx, grad_output):
        x, g, = ctx.saved_tensors
        w_mask_0 = (x > 0).float() * (g * -grad_output > 0).float() * g.abs()
        w_mask_1 = (x < 0).float() * (g * -grad_output < 0).float() * g.abs()
        scale = 10000
        w_max = (1 + scale * (w_mask_0 + w_mask_1)).max()

        grad_scale = grad_output * (1 + scale * (w_mask_0 + w_mask_1))
        return grad_scale / w_max, None

class Interval_aware_momentum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, grad):
        ctx.save_for_backward(x, g, grad)
        return x
    @staticmethod
    def backward(ctx, grad_output):
        x, g, grad_momentum, = ctx.saved_tensors
        w_mask_0 = (x > 0).float() * (g * grad_momentum > 0).float() * g.abs()
        w_mask_1 = (x < 0).float() * (g * grad_momentum < 0).float() * g.abs()
        scale = 10000

        grad_scale = grad_output * (1 + scale * (w_mask_0 + w_mask_1))
        return grad_scale, None, None

class RoundFunction_distribution(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n, distribute_grad):
        ctx.save_for_backward(distribute_grad)
        return torch.round(x * n) / n
    @staticmethod
    def backward(ctx, grad_output):
        distribute_grad, = ctx.saved_tensors
        grad_out = grad_output + (1 * distribute_grad)
        diff_grad = ((grad_out * grad_output) > 0).float()
        grad_out = diff_grad * grad_out
        # grad_input = grad_output.clone()
        return grad_out, None, None

class RoundFunction_distribution_scale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n, distribute_grad):
        ctx.save_for_backward(distribute_grad)
        return torch.round(x * n) / n
    @staticmethod
    def backward(ctx, grad_output):
        distribute_grad, = ctx.saved_tensors
        same = ((grad_output * distribute_grad) > 0).float()
        alpha = 1000
        grad_scale = (1 + (alpha * ((same * distribute_grad.abs()) + ((same - 1) * distribute_grad.abs()))))
        grad_out = grad_output * grad_scale
        # grad_input = grad_output.clone()
        return grad_out, None, None

class Gradient_zero(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, zero_mask):
        ctx.save_for_backward(zero_mask)
        return x
    @staticmethod
    def backward(ctx, grad_output):
        zero_mask, = ctx.saved_tensors
        return grad_output * zero_mask, None

class Gradient_same(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scaling):
        ctx.save_for_backward(scaling)
        return x
    @staticmethod
    def backward(ctx, grad_output):
        scaling, = ctx.saved_tensors
        return grad_output / scaling, None

class Gradient_scale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scaling_gradient):
        ctx.save_for_backward(scaling_gradient)
        return x
    @staticmethod
    def backward(ctx, grad_output):
        scaling_gradient, = ctx.saved_tensors
        return grad_output * scaling_gradient, None

class DSQConv_a(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                momentum = 0.1, num_bit = 3, QInput = True, bSetQ = True):
        super(DSQConv_a, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bit = num_bit
        self.quan_input = QInput
        self.bit_range = 2**self.num_bit -1
        self.is_quan = bSetQ
        self.temp = -1

        self.w_mask_0 = None
        self.w_mask_1 = None
        self.w_mask_2 = None
        self.w_mask_3 = None

        self.w_weight_0 = None
        self.w_weight_1 = None
        self.w_weight_2 = None
        self.w_weight_3 = None

        self.w_num_0 = None
        self.w_num_1 = None
        self.w_num_2 = None
        self.w_num_3 = None 

        self.a_mask_0 = None
        self.a_mask_1 = None
        self.a_mask_2 = None
        self.a_mask_3 = None

        self.a_weight_0 = None
        self.a_weight_1 = None
        self.a_weight_2 = None
        self.a_weight_3 = None

        self.a_num_0 = None
        self.a_num_1 = None
        self.a_num_2 = None
        self.a_num_3 = None

        self.a0_mean = None
        self.a1_mean = None
        self.a2_mean = None
        self.a3_mean = None

        self.a_ratio = torch.tensor(1.0)
        self.a0_ratio = torch.tensor(1.0)
        self.a1_ratio = torch.tensor(1.0)
        self.a2_ratio = torch.tensor(1.0)
        self.a3_ratio = torch.tensor(1.0)

        self.w_ratio = torch.tensor(1.0)
        self.w0_ratio = torch.tensor(1.0)
        self.w1_ratio = torch.tensor(1.0)
        self.w2_ratio = torch.tensor(1.0)
        self.w3_ratio = torch.tensor(1.0)

        self.q_error_activation = None
        self.q_error_activation_0 = None
        self.q_error_activation_1 = None
        self.q_error_activation_2 = None
        self.q_error_activation_3 = None

        self.pre_a0_ = torch.tensor(1.0)
        self.pre_a1_ = torch.tensor(1.0)
        self.pre_a2_ = torch.tensor(1.0)
        self.pre_a3_ = torch.tensor(1.0)

        self.pre_w0_ = None
        self.pre_w1_ = None
        self.pre_w2_ = None
        self.pre_w3_ = None
        
        self.q_error_weight = None
        self.q_error_weight_0 = None
        self.q_error_weight_1 = None
        self.q_error_weight_2 = None
        self.q_error_weight_3 = None

        self.weight_buff = None
        self.weight_clamp = None
        self.act_buff = None
        self.act_clamp = None

        self.weight_norm = None
        self.act_norm = None
        self.act_in = None

        self.weight_out = None
        self.act_out = None

        self.w_grad = torch.tensor(0.0)
        self.a_grad = torch.tensor(0.0)

        self.w_grad_real = torch.tensor(0.0)    

        self.training_ratio = 0

        if self.is_quan:
            # using int32 max/min as init and backprogation to optimization
            # Weight
            self.sW = nn.Parameter(data = torch.tensor(8).float())
            self.register_buffer('init', torch.tensor(1).float())
            self.beta = nn.Parameter(data = torch.tensor(0.2).float())

            # Activation input
            if self.quan_input:
                self.uA = nn.Parameter(data = torch.tensor(2 **31 - 1).float())

                # for activation parameter for beta
                self.a0_ = nn.Parameter(data = torch.tensor(1).float())
                self.a1_ = nn.Parameter(data = torch.tensor(1).float())
                self.a2_ = nn.Parameter(data = torch.tensor(1).float())
                self.a3_ = nn.Parameter(data = torch.tensor(1).float())

                self.w0_ = nn.Parameter(data = torch.tensor(1).float())    
                self.w1_ = nn.Parameter(data = torch.tensor(1).float())
                self.w2_ = nn.Parameter(data = torch.tensor(1).float())
                self.w3_ = nn.Parameter(data = torch.tensor(1).float())
                
                # for weight_parameter
                self.w0 = nn.Parameter(data = torch.tensor(1).float())    
                self.w1 = nn.Parameter(data = torch.tensor(1).float())
                self.w2 = nn.Parameter(data = torch.tensor(1).float())
                self.w3 = nn.Parameter(data = torch.tensor(1).float())
                

                # for activation parameter
                self.a0 = nn.Parameter(data = torch.tensor(1).float())
                self.a1 = nn.Parameter(data = torch.tensor(1).float())
                self.a2 = nn.Parameter(data = torch.tensor(1).float())
                self.a3 = nn.Parameter(data = torch.tensor(1).float())
                


    def quantizer(self, x, n):
        return torch.round(x * n) / n

    def quantizer_real(self, x, s, bit):
        out = (x / torch.abs(s)).detach()
        out = (out + 1) * 0.5
        out_c = torch.clamp(out, min=0, max=1)
        return out_c


    def w_quan(self, x, s):

        bit = 2
        bit_range = (2 ** bit) - 1
        # bit_range = 1


        out = (x / torch.abs(s))
        out_b = (out + 1) * 0.5

        out_f = torch.clamp(out_b, min=0, max=1)        
        out = RoundFunction.apply(out_f, bit_range)
        out_floor = (out_f * bit_range).floor() / bit_range

        self.w_num_0 = (out == (0/bit_range))
        self.w_num_1 = (out == (1/bit_range))
        self.w_num_2 = (out == (2/bit_range))
        self.w_num_3 = (out == (3/bit_range))

        decay = (torch.cos(torch.tensor(self.training_ratio) * math.pi) + 1) / 2
        ratio = 1.5
        momentum = 0.01 * decay

        if self.training:

            if self.w0_ == 1:
                w0_mean = out_b[self.w_num_0].mean().detach()
                w1_mean = out_b[self.w_num_1].mean().detach()
                w2_mean = out_b[self.w_num_2].mean().detach()
                w3_mean = out_b[self.w_num_3].mean().detach()    

                self.w0_.data = torch.tensor(0.0).cuda()
                self.w1_.data = torch.tensor(0.0).cuda()
                self.w2_.data = torch.tensor(0.0).cuda()
                self.w3_.data = torch.tensor(0.0).cuda()
                # self.w0_.data = bit_range * (w0_mean - (0/bit_range))
                # self.w1_.data = bit_range * (w1_mean - (1/bit_range))
                # self.w2_.data = bit_range * (w2_mean - (2/bit_range))
                # self.w3_.data = bit_range * (w3_mean - (3/bit_range))
            
            # else:
            #     self.w0_.data = (1- (0.001)) * self.w0_.data + 0.001 * (w0_mean - (0/bit_range))
            #     self.w1_.data = (1- (0.001)) * self.w1_.data + 0.001 * (w1_mean - (1/bit_range))
            #     self.w2_.data = (1- (0.001)) * self.w2_.data + 0.001 * (w2_mean - (2/bit_range))
            #     self.w3_.data = (1- (0.001)) * self.w3_.data + 0.001 * (w3_mean - (3/bit_range))

            # if self.w_ratio > ratio:
            
            # if False:
            #     self.w0_.data = (1- (momentum)) * self.w0_.data + momentum * (w0_mean - (0/bit_range))
            #     self.w1_.data = (1- (momentum)) * self.w1_.data + momentum * (w1_mean - (1/bit_range))
            #     self.w2_.data = (1- (momentum)) * self.w2_.data + momentum * (w2_mean - (2/bit_range))
            #     self.w3_.data = (1- (momentum)) * self.w3_.data + momentum * (w3_mean - (3/bit_range))

        # w_num_0 = (out_floor == (0/bit_range)) + (out_floor == (1/bit_range))
        w_num_0 = (out_floor == (0/bit_range))
        w_num_1 = (out_floor == (1/bit_range))
        w_num_2 = (out_floor == (2/bit_range)) + (out_floor == (3/bit_range))

        # scaling_gradient_0 = w_num_0.float() * ((self.w1 * (1/bit_range) + self.w1_/bit_range) - (self.w0 * (0/bit_range) + self.w0_/bit_range)) / (1/bit_range)
        # scaling_gradient_1 = w_num_1.float() * ((self.w2 * (2/bit_range) + self.w2_/bit_range) - (self.w1 * (1/bit_range) + self.w1_/bit_range)) / (1/bit_range)
        # scaling_gradient_2 = w_num_2.float() * ((self.w3 * (3/bit_range) + self.w3_/bit_range) - (self.w2 * (2/bit_range) + self.w2_/bit_range)) / (1/bit_range)

        # scaling_gradient_0 = w_num_0.float() * ((self.w1 * (1/bit_range) + self.w1_) - (self.w0 * (0/bit_range) + self.w0_)) / (1/bit_range)
        # scaling_gradient_1 = w_num_1.float() * ((self.w2 * (2/bit_range) + self.w2_) - (self.w1 * (1/bit_range) + self.w1_)) / (1/bit_range)
        # scaling_gradient_2 = w_num_2.float() * ((self.w3 * (3/bit_range) + self.w3_) - (self.w2 * (2/bit_range) + self.w2_)) / (1/bit_range)

        # scaling_gradient = scaling_gradient_0 
        # scaling_gradient = scaling_gradient_0 + scaling_gradient_1 + scaling_gradient_2

        # scaling = (self.w_num_0 * self.w0) + (self.w_num_1 * self.w1)
        # scaling_beta = (self.w_num_0 * self.w0_) + (self.w_num_1 * self.w1_)
        # scaling = (self.w_num_0 * self.w0) + (self.w_num_1 * self.w1) + (self.w_num_2 * self.w2) + (self.w_num_3 * self.w3)

        w0_ = self.w_num_0 * self.w0_
        weight_0 = (torch.exp(-(bit_range*(out_f - (0/bit_range))) ** 2) - torch.exp(torch.tensor(-1.)))
        w0_ = Gradient_scale.apply(w0_, weight_0)

        w1_ = self.w_num_1 * self.w1_
        weight_1 = (torch.exp(-(bit_range*(out_f - (1/bit_range))) ** 2) - torch.exp(torch.tensor(-1.)))
        w1_ = Gradient_scale.apply(w1_, weight_1)

        w2_ = self.w_num_2 * self.w2_
        weight_2 = (torch.exp(-(bit_range*(out_f - (2/bit_range))) ** 2) - torch.exp(torch.tensor(-1.)))
        w2_ = Gradient_scale.apply(w2_, weight_2)

        w3_ = self.w_num_3 * self.w3_
        weight_3 = (torch.exp(-(bit_range*(out_f - (3/bit_range))) ** 2) - torch.exp(torch.tensor(-1.)))
        w3_ = Gradient_scale.apply(w3_, weight_3)

        scaling_beta = w0_ + w1_ + w2_ + w3_
        # scaling_beta = (self.w_num_0 * self.w0_) + (self.w_num_1 * self.w1_) + (self.w_num_2 * self.w2_) + (self.w_num_3 * self.w3_)
        scaling_beta = scaling_beta / bit_range

        # out = Gradient_scale.apply(out, scaling_gradient)
        # out_q = Gradient_same.apply(out, scaling)
        out_q = out
        out = out_q + scaling_beta

        self.q_error_weight = ((out_b - out) ** 2).mean()
        # self.q_error_weight_0 = ((out_b - out)[self.w_num_0] ** 2).mean()
        # self.q_error_weight_1 = ((out_b - out)[self.w_num_1] ** 2).mean()

        self.w_ratio = self.q_error_weight / ((out_b - out_q) ** 2).mean()
        # self.w0_ratio = self.q_error_weight_0 / ((out_b - out_q)[self.w_num_0] ** 2).mean()
        # self.w1_ratio = self.q_error_weight_1 / ((out_b - out_q)[self.w_num_1] ** 2).mean()

        out_real = (out - 0.5) * 2

        return out_real

    def a_quan(self, x, u, l):
        delta = (u - l)

        out_b = (x - l) / delta

        bit = 2
        bit_range = (2 ** bit) - 1
        # bit_range = 1

        out_f = torch.clamp(out_b, min=0, max=1)
        out = RoundFunction.apply(out_f, bit_range)
        
        self.a_num_0 = (out == (0/bit_range))
        self.a_num_1 = (out == (1/bit_range))
        self.a_num_2 = (out == (2/bit_range))
        self.a_num_3 = (out == (3/bit_range))

        out_floor = (out_f * bit_range).floor() / bit_range

        decay = (torch.cos(torch.tensor(self.training_ratio) * math.pi) + 1) / 2
        ratio = 1.5
        momentum = 0.01 * decay

        if self.training:

            alpha = 1.0

            if self.a0_ == 1:
                a0_mean = out_b[self.a_num_0].mean().detach()
                a1_mean = out_b[self.a_num_1].mean().detach()
                a2_mean = out_b[self.a_num_2].mean().detach()
                a3_mean = out_b[self.a_num_3].mean().detach()

                self.a0_.data = torch.tensor(0.0).cuda()
                self.a1_.data = torch.tensor(0.0).cuda()
                self.a2_.data = torch.tensor(0.0).cuda()
                self.a3_.data = torch.tensor(0.0).cuda()
                # self.a0_.data = bit_range * (a0_mean - (0/bit_range))
                # self.a1_.data = bit_range * (a1_mean - (1/bit_range))
                # self.a2_.data = bit_range * (a2_mean - (2/bit_range))
                # self.a3_.data = bit_range * (a3_mean - (3/bit_range))

            # if self.a_ratio > 1:
            #     self.a0_.data = (self.a0_.data - self.pre_a0_.data) / self.a_ratio.data + self.pre_a0_.data
            #     self.a1_.data = (self.a1_.data - self.pre_a1_.data) / self.a_ratio.data + self.pre_a1_.data
            #     self.a2_.data = (self.a2_.data - self.pre_a2_.data) / self.a_ratio.data + self.pre_a2_.data
            #     self.a3_.data = (self.a3_.data - self.pre_a3_.data) / self.a_ratio.data + self.pre_a3_.data

            # if False:
            #     self.a0_.data = (1- (momentum)) * self.a0_.data + momentum * (a0_mean - (0/bit_range))
            #     self.a1_.data = (1- (momentum)) * self.a1_.data + momentum * (a1_mean - (1/bit_range))
            #     self.a2_.data = (1- (momentum)) * self.a2_.data + momentum * (a2_mean - (2/bit_range))
            #     self.a3_.data = (1- (momentum)) * self.a3_.data + momentum * (a3_mean - (3/bit_range))

        # a_num_0 = (out_floor == (0/bit_range)) + (out_floor == (1/bit_range))
        a_num_0 = (out_floor == (0/bit_range))
        a_num_1 = (out_floor == (1/bit_range))
        a_num_2 = (out_floor == (2/bit_range)) + (out_floor == (3/bit_range))

        # scaling_gradient_0 = a_num_0.float() * ((self.a1 * (1/bit_range) + self.a1_/bit_range) - (self.a0 * (0/bit_range) + self.a0_/bit_range)) / (1/bit_range)
        # scaling_gradient_1 = a_num_1.float() * ((self.a2 * (2/bit_range) + self.a2_/bit_range) - (self.a1 * (1/bit_range) + self.a1_/bit_range)) / (1/bit_range)
        # scaling_gradient_2 = a_num_2.float() * ((self.a3 * (3/bit_range) + self.a3_/bit_range) - (self.a2 * (2/bit_range) + self.a2_/bit_range)) / (1/bit_range)

        # scaling_gradient_0 = a_num_0.float() * ((self.a1 * (1/bit_range) + self.a1_) - (self.a0 * (0/bit_range) + self.a0_)) / (1/bit_range)
        # scaling_gradient_1 = a_num_1.float() * ((self.a2 * (2/bit_range) + self.a2_) - (self.a1 * (1/bit_range) + self.a1_)) / (1/bit_range)
        # scaling_gradient_2 = a_num_2.float() * ((self.a3 * (3/bit_range) + self.a3_) - (self.a2 * (2/bit_range) + self.a2_)) / (1/bit_range)

        # scaling_gradient = scaling_gradient_0
        # scaling_gradient = scaling_gradient_0 + scaling_gradient_1 + scaling_gradient_2

        # scaling = (self.a_num_0 * self.a0) + (self.a_num_1 * self.a1)
        # scaling_beta = (self.a_num_0 * self.a0_) + (self.a_num_1 * self.a1_)

        a0_ = self.a_num_0 * self.a0_
        weight_0 = (torch.exp(-(bit_range*(out_f - (0/bit_range))) ** 2) - torch.exp(torch.tensor(-1.)))
        a0_ = Gradient_scale.apply(a0_, weight_0)

        a1_ = self.a_num_1 * self.a1_
        weight_1 = (torch.exp(-(bit_range*(out_f - (1/bit_range))) ** 2) - torch.exp(torch.tensor(-1.)))
        a1_ = Gradient_scale.apply(a1_, weight_1)

        a2_ = self.a_num_2 * self.a2_
        weight_2 = (torch.exp(-(bit_range*(out_f - (2/bit_range))) ** 2) - torch.exp(torch.tensor(-1.)))
        a2_ = Gradient_scale.apply(a2_, weight_2)

        a3_ = self.a_num_3 * self.a3_
        weight_3 = (torch.exp(-(bit_range*(out_f - (3/bit_range))) ** 2) - torch.exp(torch.tensor(-1.)))
        a3_ = Gradient_scale.apply(a3_, weight_3)

        # scaling = (self.a_num_0 * self.a0) + (self.a_num_1 * self.a1) + (self.a_num_2 * self.a2) + (self.a_num_3 * self.a3)
        scaling_beta = a0_ + a1_ + a2_ + a3_
        # scaling_beta = (self.a_num_0 * self.a0_) + (self.a_num_1 * self.a1_) + (self.a_num_2 * self.a2_) + (self.a_num_3 * self.a3_)
        scaling_beta = scaling_beta / bit_range

        # out = Gradient_scale.apply(out, scaling_gradient)
        # out_q = Gradient_same.apply(out, scaling)
        out_q = out
        out = out_q + scaling_beta

        self.q_error_activation = ((out_b - out) ** 2).mean()
        # self.q_error_activation_0 = ((out_b - out)[self.a_num_0] ** 2).mean()
        # self.q_error_activation_1 = ((out_b - out)[self.a_num_1] ** 2).mean()
        
        ## Savign the previous value
        self.pre_a0_ = torch.tensor(self.a0_.item()).cuda()
        self.pre_a1_ = torch.tensor(self.a1_.item()).cuda()
        self.pre_a2_ = torch.tensor(self.a2_.item()).cuda()
        self.pre_a3_ = torch.tensor(self.a3_.item()).cuda()
        
        # self.q_error_activation_2 = ((out_b - out)[self.a_num_2] ** 2).mean()
        # self.q_error_activation_3 = ((out_b - out)[self.a_num_3] ** 2).mean()

        self.a_ratio = self.q_error_activation / ((out_b - out_q) ** 2).mean()
        # self.a0_ratio = self.q_error_activation_0 / ((out_b - out_q)[self.a_num_0] ** 2).mean()
        # self.a1_ratio = self.q_error_activation_1 / ((out_b - out_q)[self.a_num_1] ** 2).mean()
        # self.a2_ratio = self.q_error_activation_2 / ((out_b - out_q)[self.a_num_2] ** 2).mean()
        # self.a3_ratio = self.q_error_activation_3 / ((out_b - out_q)[self.a_num_3] ** 2).mean()

        return out

    def forward(self, x):

        if self.is_quan:
            if self.init == 1:
                print(self.init)
                self.sW.data = self.weight.std() * 3
                self.uA.data = (x.std() / math.sqrt(1 - 2/math.pi)) * 3

            Qweight = self.w_quan(self.weight, self.sW)
            Qbias = self.bias

            # Input(Activation)
            Qactivation = x

            if self.quan_input:
                Qactivation = self.a_quan(x, self.uA, 0)

            if self.init == 1:
                # print(self.init)
                self.init = torch.tensor(0)
                q_output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
                ori_output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

                self.beta.data = torch.mean(torch.abs(ori_output)) / \
                                 torch.mean(torch.abs(q_output))

            output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
            output = torch.abs(self.beta) * output

        else:
            output =  F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

        return output

# class DSQConv_a(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
#                 momentum = 0.1, num_bit = 3, QInput = True, bSetQ = True):
#         super(DSQConv_a, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
#         self.num_bit = num_bit
#         self.quan_input = QInput
#         self.bit_range = 2**self.num_bit -1
#         self.is_quan = bSetQ
#         self.temp = -1

#         self.w_mask_0 = None
#         self.w_mask_1 = None
#         self.w_mask_2 = None
#         self.w_mask_3 = None

#         self.w_weight_0 = None
#         self.w_weight_1 = None
#         self.w_weight_2 = None
#         self.w_weight_3 = None

#         self.w_num_0 = None
#         self.w_num_1 = None
#         self.w_num_2 = None
#         self.w_num_3 = None 

#         self.a_mask_0 = None
#         self.a_mask_1 = None
#         self.a_mask_2 = None
#         self.a_mask_3 = None

#         self.a_weight_0 = None
#         self.a_weight_1 = None
#         self.a_weight_2 = None
#         self.a_weight_3 = None

#         self.a_num_0 = None
#         self.a_num_1 = None
#         self.a_num_2 = None
#         self.a_num_3 = None

#         self.w_grad_0 = torch.tensor(0.0)
#         self.w_grad_1 = torch.tensor(0.0)
#         self.w_grad_2 = torch.tensor(0.0)
#         self.w_grad_3 = torch.tensor(0.0)

#         self.a_grad_0 = torch.tensor(0.0)
#         self.a_grad_1 = torch.tensor(0.0)
#         self.a_grad_2 = torch.tensor(0.0)
#         self.a_grad_3 = torch.tensor(0.0)

#         self.weight_buff = None
#         self.weight_clamp = None
#         self.act_buff = None
#         self.act_clamp = None

#         self.weight_norm = None
#         self.act_norm = None
#         self.act_in = None

#         self.weight_out = None
#         self.act_out = None

#         self.training_ratio = 0

#         if self.is_quan:
#             # using int32 max/min as init and backprogation to optimization
#             # Weight
#             self.sW = nn.Parameter(data = torch.tensor(8).float())
#             self.register_buffer('init', torch.tensor([1]).float())
#             self.beta = nn.Parameter(data = torch.tensor(0.2).float())

#             # Activation input
#             if self.quan_input:
#                 self.uA = nn.Parameter(data = torch.tensor(2 **31 - 1).float())

#     def quantizer(self, x, n):
#         return torch.round(x * n) / n

#     def quantizer_real(self, x, s, bit):
#         out = (x / torch.abs(s)).detach()
#         out = (out + 1) * 0.5
#         out_c = torch.clamp(out, min=0, max=1)
#         return out_c


#     def w_quan(self, x, s):

#         bit = 2
#         bit_range = (2 ** bit) - 1

#         out_c = self.quantizer_real(x, s, bit)
#         out_compare = (out_c * bit_range).floor() / bit_range
        
#         self.w_mask_0 = (out_compare == out_compare) 
#         # self.w_mask_1 = (out_compare == (0/bit_range))
#         self.w_mask_1 = ((out_compare == (0/bit_range)).float() - (out_c == 0).float()).bool()
#         self.w_mask_2 = (out_compare == (1/bit_range))
#         self.w_mask_3 = (out_compare == (2/bit_range))

#         distribute_grad = (self.w_mask_1 * self.w_grad_1) + (self.w_mask_2 * self.w_grad_2) + (self.w_mask_3 * self.w_grad_3)

#         self.weight_buff = x
#         # self.weight_buff = Decay_aware.apply(x, torch.abs(s))
#         # self.weight_buff = Distribution.apply(self.weight_buff, distribute_grad)

#         if self.training:
#             self.weight_buff.retain_grad()

#         out = (self.weight_buff / torch.abs(s))
#         out = (out + 1) * 0.5

#         out = torch.clamp(out, min=0, max=1)

#         self.weight_clamp = ((2*out) - 1) * torch.abs(s)
        
#         out = RoundFunction.apply(out, bit_range)
#         # out = EWGS_discretizer.apply(out, bit_range)

#         self.w_num_0 = (out == (0/bit_range))
#         self.w_num_1 = (out == (1/bit_range))
#         self.w_num_2 = (out == (2/bit_range))
#         self.w_num_3 = (out == (3/bit_range))

#         out_real = (2 * out) - 1

#         return out_real

#     def a_quan(self, x, u, l):
#         delta = (u - l)

#         self.act_in = x

#         if self.training:
#             self.act_in.retain_grad()

#         out = (self.act_in - l) / delta

#         bit = 2
#         bit_range = (2 ** bit) - 1
#         # bit_range = 2

#         self.act_norm = torch.clamp(out, min=0, max=1)

#         self.act_clamp = u * self.act_norm

#         if self.training:
#             self.act_norm.retain_grad()
        
#         out = self.quantizer(self.act_norm, bit_range)

#         out_compare = (self.act_norm * bit_range).floor() / bit_range
#         # level-4
#         # self.a_mask_0 = ((out_compare == out_compare).float() - (self.act_norm == 0).float() - (self.act_norm == 1).float()).bool()
#         # self.a_mask_1 = ((out_compare == (0/bit_range)).float() - (self.act_norm == 0).float()).bool()
#         # self.a_mask_2 = (out_compare == (1/bit_range))
#         # self.a_mask_3 = ((out_compare == (2/bit_range)).float() - (self.act_norm == 1).float()).bool()

#         self.a_mask_0 = (out_compare == out_compare)
#         # self.a_mask_1 = (out_compare == (0/bit_range))
#         self.a_mask_1 = ((out_compare == (0/bit_range)).float() - (self.act_norm == 0).float()).bool()
#         self.a_mask_2 = (out_compare == (1/bit_range))
#         self.a_mask_3 = (out_compare == (2/bit_range))

#         distribute_grad = (self.a_mask_1 * self.a_grad_1) + (self.a_mask_2 * self.a_grad_2) + (self.a_mask_3 * self.a_grad_3)

#         # self.act_buff = Distribution.apply(self.act_in, distribute_grad)
#         self.act_buff = self.act_in
#         if self.training:
#             self.act_buff.retain_grad()

#         out = (self.act_buff - l) / delta
#         out = torch.clamp(out, min=0, max=1)
#         out = RoundFunction.apply(out, bit_range)

#         self.a_num_0 = (out == (0/bit_range))
#         self.a_num_1 = (out == (1/bit_range))
#         self.a_num_2 = (out == (2/bit_range))
#         self.a_num_3 = (out == (3/bit_range))

#         return out

#     def forward(self, x):

#         if self.is_quan:
#             if self.init == 1:
#                 print(self.init)
#                 self.sW.data = self.weight.std() * 3
#                 self.uA.data = (x.std() / math.sqrt(1 - 2/math.pi)) * 3

#             Qweight = self.w_quan(self.weight, self.sW)
#             Qbias = self.bias

#             # Input(Activation)
#             Qactivation = x

#             if self.quan_input:
#                 Qactivation = self.a_quan(x, self.uA, 0)

#             if self.init == 1:
#                 # print(self.init)
#                 self.init = torch.tensor(0)
#                 q_output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
#                 ori_output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

#                 self.beta.data = torch.mean(torch.abs(ori_output)) / \
#                                  torch.mean(torch.abs(q_output))

#             output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
#             output = torch.abs(self.beta) * output

#         else:
#             output =  F.conv2d(x, self.weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)

#         return output


# class DSQConv_a(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
#                 momentum = 0.1, num_bit = 3, QInput = True, bSetQ = True):
#         super(DSQConv_a, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
#         self.num_bit = num_bit
#         self.quan_input = QInput
#         self.bit_range = 2**self.num_bit -1
#         self.is_quan = bSetQ
#         self.temp = -1

#         self.w_mask_0 = None
#         self.w_mask_1 = None
#         self.w_mask_2 = None
#         self.w_mask_3 = None

#         self.w_weight_0 = None
#         self.w_weight_1 = None
#         self.w_weight_2 = None
#         self.w_weight_3 = None

#         self.w_num_0 = None
#         self.w_num_1 = None
#         self.w_num_2 = None
#         self.w_num_3 = None 

#         self.a_mask_0 = None
#         self.a_mask_1 = None
#         self.a_mask_2 = None
#         self.a_mask_3 = None

#         self.a_weight_0 = None
#         self.a_weight_1 = None
#         self.a_weight_2 = None
#         self.a_weight_3 = None

#         self.a_num_0 = None
#         self.a_num_1 = None
#         self.a_num_2 = None
#         self.a_num_3 = None

#         self.weight_buff = None
#         self.weight_clamp = None
#         self.act_buff = None
#         self.act_clamp = None

#         self.weight_norm = None
#         self.act_norm = None
#         self.act_in = None

#         self.weight_out = None
#         self.act_out = None

#         self.w_grad = torch.tensor(0.0)
#         self.a_grad = torch.tensor(0.0)

#         self.w_grad_real = torch.tensor(0.0)    

#         self.training_ratio = 0

#         if self.is_quan:
#             # using int32 max/min as init and backprogation to optimization
#             # Weight
#             self.sW = nn.Parameter(data = torch.tensor(8).float())
#             self.register_buffer('init', torch.tensor([1]).float())
#             self.beta = nn.Parameter(data = torch.tensor(0.2).float())

#             # Activation input
#             if self.quan_input:
#                 self.uA = nn.Parameter(data = torch.tensor(2 **31 - 1).float())

#     def quantizer(self, x, n):
#         return torch.round(x * n) / n

#     def quantizer_real(self, x, s, bit):
#         out = (x / torch.abs(s)).detach()
#         out = (out + 1) * 0.5
#         out_c = torch.clamp(out, min=0, max=1)
#         return out_c


#     def w_quan(self, x, s):

#         bit = 3
#         bit_range = (2 ** bit) - 1
#         bit_range = 2

#         out_c = self.quantizer_real(x, s, bit)
#         out_compare = (out_c * bit_range).floor() / bit_range
        
#         self.w_mask_0 = (out_compare == out_compare) 
#         self.w_mask_1 = (x < 0)
#         self.w_mask_2 = (x > 0)

#         # x = Interval_aware.apply(x, self.w_grad / self.w_mask_0.numel())
#         # x = Interval_aware_momentum.apply(x, self.w_grad / self.w_mask_0.numel(), self.w_grad_real * (torch.cos(torch.tensor(self.training_ratio) * math.pi) + 1))
#         # x = Interval_aware_momentum.apply(x, self.w_grad / self.w_mask_0.numel(), self.w_grad_real)
#         self.weight_buff = x

        

#         if self.training:
#             self.weight_buff.retain_grad()

#         out = (self.weight_buff / torch.abs(s))
#         out = (out + 1) * 0.5

#         out = torch.clamp(out, min=0, max=1)

#         self.weight_clamp = ((2*out) - 1) * torch.abs(s)
        
#         out = RoundFunction.apply(out, bit_range)
#         # out = EWGS_discretizer.apply(out, bit_range)

#         self.w_num_0 = (out == (0/bit_range))
#         self.w_num_1 = (out == (1/bit_range))
#         self.w_num_2 = (out == (2/bit_range))
#         self.w_num_3 = (out == (3/bit_range))

#         out_real = (2 * out) - 1

#         return out_real

#     def a_quan(self, x, u, l):
#         delta = (u - l)

#         self.act_in = x

#         if self.training:
#             self.act_in.retain_grad()

#         out = (self.act_in - l) / delta

#         bit = 3
#         bit_range = (2 ** bit) - 1
#         bit_range = 2

#         self.act_norm = torch.clamp(out, min=0, max=1)

#         self.act_clamp = u * self.act_norm

#         if self.training:
#             self.act_norm.retain_grad()
        
#         out = self.quantizer(self.act_norm, bit_range)

#         out_compare = (self.act_norm * bit_range).floor() / bit_range

#         self.a_mask_0 = (out_compare == out_compare)
#         self.a_mask_1 = (out_compare == out_compare)
#         self.act_buff = self.act_in

#         if self.training:
#             self.act_buff.retain_grad()

#         out = (self.act_buff - l) / delta
#         out = torch.clamp(out, min=0, max=1)
#         out = RoundFunction.apply(out, bit_range)

#         self.a_num_0 = (out == (0/bit_range))
#         self.a_num_1 = (out == (1/bit_range))
#         self.a_num_2 = (out == (2/bit_range))
#         self.a_num_3 = (out == (3/bit_range))

#         return out

#     def forward(self, x):

#         if self.is_quan:
#             if self.init == 1:
#                 print(self.init)
#                 self.sW.data = self.weight.std() * 3
#                 self.uA.data = (x.std() / math.sqrt(1 - 2/math.pi)) * 3

#             Qweight = self.w_quan(self.weight, self.sW)
#             Qbias = self.bias

#             # Input(Activation)
#             Qactivation = x

#             if self.quan_input:
#                 Qactivation = self.a_quan(x, self.uA, 0)

#             if self.init == 1:
#                 # print(self.init)
#                 self.init = torch.tensor(0)
#                 q_output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
#                 ori_output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

#                 self.beta.data = torch.mean(torch.abs(ori_output)) / \
#                                  torch.mean(torch.abs(q_output))

#             output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
#             output = torch.abs(self.beta) * output

#         else:
#             output =  F.conv2d(x, self.weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)

#         return output