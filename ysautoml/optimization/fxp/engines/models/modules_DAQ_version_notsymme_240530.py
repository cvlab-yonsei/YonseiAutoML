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

forward_bit = 5
backward_bit = 32

class absol(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.abs()

    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = torch.sign(input)
        grad_input = grad_input + 1
        grad_input = ((grad_input+1e-6)/2).round()
        grad_input = (2*grad_input) - 1
        return grad_output * grad_input

class grad_quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_bits = 8, init=True, clip_ratio=1, grad_max=1):
        ctx.num_bits = num_bits
        ctx.init = init
        ctx.clip_ratio = clip_ratio
        ctx.grad_max = grad_max
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits != 32:
            alpha = ctx.clip_ratio
            # if ctx.grad_max < 0:
            #     grad_max = grad_output.abs().max()
            # else:
            #     grad_max = ctx.grad_max
            grad_max = grad_output.abs().max()
            # alpha = 1.0
            # min_value = -float(grad_output.abs().max()) * alpha
            # max_value = float(grad_output.abs().max()) * alpha

            min_value = -grad_max * alpha
            max_value = grad_max * alpha

            if min_value == 0:
                min_value = min_value - 1e-8
                max_value = max_value + 1e-8

            qmin = 0.
            qmax = 2. ** ctx.num_bits - 2.

            scale = (max_value - min_value) / (qmax - qmin)

            grad_output.add_(-min_value).div_(scale).add_(qmin)
            noise = grad_output.new(grad_output.shape).uniform_(-0.5, 0.5)
            grad_output.add_(noise)
            grad_output.clamp_(qmin,qmax).round_() # quatnize

            grad_output.add_(-qmin).mul_(scale).add_(min_value) # dequantize

        grad_input = grad_output.clone()

        return grad_input, None, None, None, None

def quantize_grad(x, num_bits, init, clip_ratio, grad_max):
    return grad_quant().apply(x, num_bits, init, clip_ratio, grad_max)

def conv2d_biprec(_input, weight, bias, stride, padding, dilation, groups, bit_grad):
    out = F.conv2d(_input, weight, bias, stride, padding, dilation, groups)
    out = quantize_grad(out, num_bits=bit_grad)
    return out

class RoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n):
        return torch.round(x * n) / n
    @staticmethod
    def backward(ctx, grad_output):
        # grad_input = grad_output.clone()
        return grad_output, None

class Conv_vis(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv_vis, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.input_feat = None

    def forward(self, x):
        self.input_feat = x
        output =  F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output

class DSQConv_a_mobile(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                momentum = 0.1, num_bit = 3, QInput = True, bSetQ = True, layer_num=0, symmetric=False):
        super(DSQConv_a_mobile, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bit = num_bit
        self.quan_input = QInput
        self.bit_range = 2**self.num_bit -1
        self.is_quan = bSetQ
        self.training_ratio = 0
        self.layer_num = layer_num
        self.quant_w = True
        self.quant_a = True
        self.output_buff = None
        self.quant_output_buff = None
        self.weight_buff = None
        self.act_buff = None
        self.pre_weight = None
        self.weight_diff = None
        self.symmetric = symmetric

        self.prev_grad = None
        self.q_start = False

        self.moment_ratio = torch.tensor(0.0)
        self.bit = torch.tensor(1.0)
        self.alpha = torch.tensor(1.0)
        # self.clip_ratio = torch.tensor(1.0)
        self.max_ratio = torch.tensor(0.0)
        self.max_std_ratio = torch.tensor(0.0)
        self.max_ratio_constant = torch.tensor(0.0)
        self.max_ratio_constant_q = torch.tensor(0.0)
        self.std_mean_ratio = torch.tensor(0.0)
        self.var_mean_ratio = torch.tensor(0.0)
        self.error_ratio = torch.tensor(0.0)

        self.std_ratio = torch.tensor(0.0)
        self.mean_ratio = torch.tensor(0.0)
        # self.grad_max = torch.tensor(0.0)

        if self.is_quan:
            # using int32 max/min as init and backprogation to optimization
            # Weight
            self.sW = nn.Parameter(data = torch.tensor(8).float())
            self.register_buffer('init', torch.tensor(1.0).float())
            self.register_buffer('clip_ratio', torch.tensor(1.0).float())
            self.register_buffer('grad_max', torch.tensor(-1.0).float())
            # self.beta = nn.Parameter(data = torch.tensor(0.2).float())

            # Activation input
            if self.quan_input:
                self.uA = nn.Parameter(data = torch.tensor(2 **31 - 1).float())

    def quantizer(self, x, n):
        return torch.round(x * n) / n

    def quantizer_real(self, x, s, bit):
        out = (x / torch.abs(s)).detach()
        out = (out + 1) * 0.5
        out_c = torch.clamp(out, min=0, max=1)
        return out_c


    def w_quan(self, x, s):
        global forward_bit

        bit = forward_bit
        bit_range = (2 ** bit) - 2

        out = (x / torch.abs(s))
        out = (out + 1) * 0.5
        out = torch.clamp(out * bit_range, min=0, max=bit_range)

        out = 2 * self.w_soft_argmax(out, 0.1, 2) - bit_range
        return out / bit_range

        # if self.quant_w:
        #     out = torch.clamp(out, min=0, max=1)
        #     out = RoundFunction.apply(out, bit_range)

        # # out = (out - 0.5) * 2
        # out = (2 * out) - 1

        # return out

    def w_soft_argmax(self, x, T, sigma):
        x_floor = x.floor()
        x = x - x_floor.detach()

        m_p = torch.exp(-absol.apply(x.unsqueeze(0).repeat(len(self.q_value.cuda()),1,1,1,1)
                                     - self.q_value.cuda()))

        # Get the kernel value
        max_value, max_idx = m_p.max(dim=0)
        max_idx = max_idx.unsqueeze(0).float().cuda()
        k_p = torch.exp(-(torch.pow(self.q_value.cuda()-max_idx, 2).float()/(sigma**2)))

        # Get the score
        score = m_p * k_p

        # Flexible temperature
        denorm = (score[0] - score[1]).abs()
        T_ori = T
        T = T / denorm
        T = T.detach()

        tmp_score = T * score

        # weighted average using the score and temperature
        prob = torch.exp(tmp_score - tmp_score.max())
        denorm2 = prob.sum(dim=0, keepdim=True)
        prob = prob / denorm2

        q_var = self.q_value.clone()
        q_var[0] = q_var[0] - (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        q_var[1] = q_var[1] + (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        output = (q_var * prob).sum(dim=0)
        output = output + x_floor

        return output

    def a_quan(self, x, u, l):
        delta = (u - l)

        out = (x - l) / delta

        global forward_bit

        bit = forward_bit

        if self.symmetric:
            bit_range = (2 ** bit) - 2
        else:
            bit_range = (2 ** bit) - 1

        if self.quant_a:
            out = torch.clamp(out, min=0, max=1)
            out = bit_range * out

            out = self.a_soft_argmax(out, 0.1, 1)

        return out / bit_range

    def a_soft_argmax(self, x, T, sigma):
        x_floor = x.floor()
        x = x - x_floor.detach()
        m_p = torch.exp(-absol.apply(x.unsqueeze(0).repeat(len(self.q_value),1,1,1,1)
                                     - self.q_value))

        # Get the kernel value
        max_value, max_idx = m_p.max(dim=0)
        max_idx = max_idx.unsqueeze(0).float().cuda()
        k_p = torch.exp(-(torch.pow(self.q_value-max_idx, 2).float()/(sigma**2)))

        # Get the score
        score = m_p * k_p

        # Flexible temperature
        denorm = (score[0] - score[1]).abs()
        T_ori = T
        T = T / denorm
        T = T.detach()

        tmp_score = T * score

        # weighted average using the score and temperature

        prob = torch.exp(tmp_score - tmp_score.max())
        denorm2 = prob.sum(dim=0, keepdim=True)
        prob = prob / denorm2

        q_var = self.q_value.clone()
        q_var[0] = q_var[0] - (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        q_var[1] = q_var[1] + (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        output = (q_var * prob).sum(dim=0)
        output = output + x_floor

        return output

    def forward(self, x):

        if self.is_quan:
            global backward_bit

            if self.init == 1:
                print(self.init)
                self.sW.data.fill_(self.weight.std() * 3)

                symmetric = ((x < 0).float().sum())

                if symmetric > 0:
                    self.symmetric = True
                else:
                    self.symmetric = False

                if self.symmetric == True:
                    self.uA.data.fill_(x.std() * 3)
                    
                else:
                    self.uA.data.fill_((x.std() / math.sqrt(1 - 2/math.pi)) * 3)

            Qweight_scale = self.w_quan(self.weight, self.sW)
            Qweight = Qweight_scale * self.sW
            Qbias = self.bias

            # Input(Activation)
            Qactivation = x

            if self.quan_input:
                if self.symmetric == True:
                    Qactivation = self.a_quan(x, self.uA, -self.uA) * self.uA
                else:
                    Qactivation = self.a_quan(x, self.uA, 0) * self.uA

            if self.init == 1:
                q_output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
                ori_output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

            layer_idx = self.layer_num % 6

            bit_grad = torch.tensor(backward_bit)
            self.bit = bit_grad

            output = F.conv2d(Qactivation, Qweight, Qbias, self.stride, self.padding, self.dilation, self.groups)

            self.quant_output_buff = output
            if self.training:
                self.quant_output_buff.retain_grad()
            
            moment = 1
            if bit_grad <= 16:
                output = quantize_grad(output, bit_grad, self.init, self.clip_ratio, self.grad_max)

            self.output_buff = output
            if self.training:
                self.output_buff.retain_grad()

            if self.init == 1:
                self.init.data.fill_(torch.tensor(0))

        else:
            output =  F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

        return output

class DSQConv_a(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                momentum = 0.1, num_bit = 3, QInput = True, bSetQ = True, layer_num=0):
        super(DSQConv_a, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bit = num_bit
        self.quan_input = QInput
        self.bit_range = 2**self.num_bit -1
        self.is_quan = bSetQ
        self.training_ratio = 0
        self.layer_num = layer_num
        self.quant_w = True
        self.quant_a = True
        self.output_buff = None
        self.quant_output_buff = None
        self.weight_buff = None
        self.act_buff = None
        self.pre_weight = None
        self.weight_diff = None

        self.prev_grad = None
        self.q_start = False
        self.q_value = torch.from_numpy(np.linspace(0,1,2))
        self.q_value = self.q_value.reshape(len(self.q_value),1,1,1,1).float().cuda()

        self.moment_ratio = torch.tensor(0.0)
        self.bit = torch.tensor(1.0)
        self.alpha = torch.tensor(1.0)
        # self.clip_ratio = torch.tensor(1.0)
        self.max_ratio = torch.tensor(0.0)
        self.max_std_ratio = torch.tensor(0.0)
        self.max_ratio_constant = torch.tensor(0.0)
        self.max_ratio_constant_q = torch.tensor(0.0)
        self.std_mean_ratio = torch.tensor(0.0)
        self.var_mean_ratio = torch.tensor(0.0)
        self.error_ratio = torch.tensor(0.0)

        self.std_ratio = torch.tensor(0.0)
        self.mean_ratio = torch.tensor(0.0)

        self.register_buffer('lr_scale', torch.tensor(1.0).float())

        # self.grad_max = torch.tensor(0.0)

        if self.is_quan:
            # using int32 max/min as init and backprogation to optimization
            # Weight
            self.sW = nn.Parameter(data = torch.tensor(8).float())
            self.register_buffer('init', torch.tensor(1.0).float())
            self.register_buffer('clip_ratio', torch.tensor(1.0).float())
            self.register_buffer('grad_max', torch.tensor(-1.0).float())
            # self.beta = nn.Parameter(data = torch.tensor(0.2).float())

            # Activation input
            if self.quan_input:
                self.uA = nn.Parameter(data = torch.tensor(2 **31 - 1).float())

    def quantizer(self, x, n):
        return torch.round(x * n) / n

    def quantizer_real(self, x, s, bit):
        out = (x / torch.abs(s)).detach()
        out = (out + 1) * 0.5
        out_c = torch.clamp(out, min=0, max=1)
        return out_c



    def w_quan(self, x, s):
        global forward_bit

        bit = forward_bit
        bit_range = (2 ** bit) - 2

        out = (x / torch.abs(s))
        out = (out + 1) * 0.5
        out = torch.clamp(out * bit_range, min=0, max=bit_range)

        out = 2 * self.w_soft_argmax(out, 2, 1) - bit_range
        return out / bit_range

        # if self.quant_w:
        #     out = torch.clamp(out, min=0, max=1)
        #     out = RoundFunction.apply(out, bit_range)

        # # out = (out - 0.5) * 2
        # out = (2 * out) - 1

        # return out

    def w_soft_argmax(self, x, T, sigma):
        x_floor = x.floor()
        x = x - x_floor.detach()

        m_p = torch.exp(-absol.apply(x.unsqueeze(0).repeat(len(self.q_value.cuda()),1,1,1,1)
                                     - self.q_value.cuda()))

        # Get the kernel value
        max_value, max_idx = m_p.max(dim=0)
        max_idx = max_idx.unsqueeze(0).float().cuda()
        k_p = torch.exp(-(torch.pow(self.q_value.cuda()-max_idx, 2).float()/(sigma**2)))

        # Get the score
        score = m_p * k_p

        # Flexible temperature
        denorm = (score[0] - score[1]).abs()
        T_ori = T
        T = T / denorm
        T = T.detach()

        tmp_score = T * score

        # weighted average using the score and temperature
        prob = torch.exp(tmp_score - tmp_score.max())
        denorm2 = prob.sum(dim=0, keepdim=True)
        prob = prob / denorm2

        q_var = self.q_value.clone().cuda()
        q_var[0] = q_var[0] - (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        q_var[1] = q_var[1] + (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        output = (q_var * prob).sum(dim=0)
        output = output + x_floor

        return output

    # def a_quan(self, x, u, l):
    #     delta = (u - l)

    #     out = (x - l) / delta

    #     global forward_bit

    #     bit = forward_bit

    #     bit_range = (2 ** bit) - 1

    #     if self.quant_a:
    #         out = torch.clamp(out, min=0, max=1)
    #         out = bit_range * out

    #         out = self.a_soft_argmax(out, 0.1, 1)

    #     return out / bit_range

    def a_quan(self, x, u, l):
        delta = (u - l)

        out = (x - l) / delta

        global forward_bit

        bit = forward_bit

        bit_range = (2 ** bit) - 1

        if self.quant_a:
            out = torch.clamp(out, min=0, max=1)
            out = RoundFunction.apply(out, bit_range)

        return out

    def a_soft_argmax(self, x, T, sigma):
        x_floor = x.floor()
        x = x - x_floor.detach()
        m_p = torch.exp(-absol.apply(x.unsqueeze(0).repeat(len(self.q_value.cuda()),1,1,1,1)
                                     - self.q_value.cuda()))

        # Get the kernel value
        max_value, max_idx = m_p.max(dim=0)
        max_idx = max_idx.unsqueeze(0).float().cuda()
        k_p = torch.exp(-(torch.pow(self.q_value.cuda()-max_idx, 2).float()/(sigma**2)))

        # Get the score
        score = m_p * k_p

        # Flexible temperature
        denorm = (score[0] - score[1]).abs()
        T_ori = T
        T = T / denorm
        T = T.detach()

        tmp_score = T * score

        # weighted average using the score and temperature

        prob = torch.exp(tmp_score - tmp_score.max())
        denorm2 = prob.sum(dim=0, keepdim=True)
        prob = prob / denorm2

        q_var = self.q_value.clone().cuda()
        q_var[0] = q_var[0] - (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        q_var[1] = q_var[1] + (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        output = (q_var * prob).sum(dim=0)
        output = output + x_floor

        return output

    def forward(self, x):

        if self.is_quan:
            global backward_bit

            if self.init == 1:
                print(self.init)
                # self.sW.data = self.weight.std() * 3
                # self.uA.data = (x.std() / math.sqrt(1 - 2/math.pi)) * 3

                self.sW.data.fill_(self.weight.std() * 3)
                self.uA.data.fill_((x.std() / math.sqrt(1 - 2/math.pi)) * 3)

            Qweight_scale = self.w_quan(self.weight, self.sW)
            Qweight = Qweight_scale * self.sW
            # Qweight = self.w_quan(self.weight, self.sW) 
            Qbias = self.bias

            # Input(Activation)
            Qactivation = x

            if self.quan_input:
                Qactivation = self.a_quan(x, self.uA, 0) * self.uA
                # Qactivation = self.a_quan(x, self.uA, 0) 

            if self.init == 1:
                # print(self.init)
                # self.pre_weight = Qweight
                q_output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
                ori_output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

                # self.beta.data = torch.mean(torch.abs(ori_output)) / \
                #                  torch.mean(torch.abs(q_output))

                # self.beta.data.fill_(ori_output.abs().mean() / q_output.abs().mean())

            # self.weight_diff = (Qweight - self.pre_weight).abs().float().mean()

            layer_idx = self.layer_num % 6

            bit_grad = torch.tensor(backward_bit)
            self.bit = bit_grad

            # self.act_buff = Qactivation
            # self.act_buff.cuda()
            # self.act_buff.requires_grad = True¡
            # self.weight_buff = Qweight
            # self.weight_buff.cuda()

            # if self.training:
            #     self.act_buff.retain_grad()

            output = F.conv2d(Qactivation, Qweight, Qbias, self.stride, self.padding, self.dilation, self.groups)
            # output = F.conv2d(self.act_buff, self.weight_buff, Qbias, self.stride, self.padding, self.dilation, self.groups)

            self.quant_output_buff = output
            if self.training:
                self.quant_output_buff.retain_grad()
                setattr(self.weight, 'lr_scale', self.lr_scale)
            
            moment = 1
            if bit_grad <= 16:
                output = quantize_grad(output, bit_grad, self.init, self.clip_ratio, self.grad_max)

            self.output_buff = output
            if self.training:
                self.output_buff.retain_grad()

            if self.init == 1:
                self.init.data.fill_(torch.tensor(0))

            # self.pre_weight = Qweight

            # output = torch.abs(self.beta) * output

        else:
            output =  F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

        return output

class DSQConv_8bit(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                momentum = 0.1, num_bit = 3, QInput = True, bSetQ = True, layer_num=0):
        super(DSQConv_8bit, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bit = num_bit
        self.quan_input = QInput
        self.bit_range = 2**self.num_bit -1
        self.is_quan = bSetQ
        self.training_ratio = 0
        self.layer_num = layer_num
        self.quant_w = True
        self.quant_a = True
        self.output_buff = None
        self.quant_output_buff = None
        self.weight_buff = None
        self.act_buff = None
        self.pre_weight = None

        self.prev_grad = None
        self.q_start = False

        self.moment_ratio = torch.tensor(0.0)
        self.bit = torch.tensor(1.0)
        self.alpha = torch.tensor(1.0)
        # self.clip_ratio = torch.tensor(1.0)
        self.max_ratio = torch.tensor(0.0)
        self.max_std_ratio = torch.tensor(0.0)
        self.max_ratio_constant = torch.tensor(0.0)
        self.std_mean_ratio = torch.tensor(0.0)
        self.var_mean_ratio = torch.tensor(0.0)
        self.error_ratio = torch.tensor(0.0)

        self.std_ratio = torch.tensor(0.0)
        self.mean_ratio = torch.tensor(0.0)

        if self.is_quan:
            # using int32 max/min as init and backprogation to optimization
            # Weight
            self.sW = nn.Parameter(data = torch.tensor(8).float())
            self.register_buffer('init', torch.tensor(1.0).float())
            self.register_buffer('clip_ratio', torch.tensor(1.0).float())
            self.register_buffer('grad_max', torch.tensor(-1.0).float())
            # self.beta = nn.Parameter(data = torch.tensor(0.2).float())

            # Activation input
            # if self.quan_input:
                # self.uA = nn.Parameter(data = torch.tensor(2 **31 - 1).float())

    def quantizer(self, x, n):
        return torch.round(x * n) / n

    def quantizer_real(self, x, s, bit):
        out = (x / torch.abs(s)).detach()
        out = (out + 1) * 0.5
        out_c = torch.clamp(out, min=0, max=1)
        return out_c


    def w_quan(self, x, s):
        global forward_bit

        bit = forward_bit
        bit = 8

        bit_range = (2 ** bit) - 2

        out = (x / torch.abs(s))
        out = (out + 1) * 0.5
        
        out = torch.clamp(out, min=0, max=1)
        out = RoundFunction.apply(out, bit_range)

        # out = (out - 0.5) * 2
        out = (2 * out) - 1

        return out

    def a_quan(self, x, u, l):
        delta = (u - l)

        out = (x - l) / delta

        global forward_bit

        bit = forward_bit
        bit = 8

        bit_range = (2 ** bit) - 1

        out = torch.clamp(out, min=0, max=1)
        out = RoundFunction.apply(out, bit_range)

        return out

    def forward(self, x):

        if self.is_quan:
            global backward_bit

            if self.init == 1:
                print(self.init)
                self.sW.data.fill_(self.weight.std() * 3)
                # self.uA.data.fill_((x.std() / math.sqrt(1 - 2/math.pi)) * 3)
                # self.uA.data.fill_(x.std() * 3)

            Qweight = self.w_quan(self.weight, self.sW) * self.sW
            Qbias = self.bias

            # Input(Activation)
            Qactivation = x

            # if self.quan_input:
            #     Qactivation = self.a_quan(x, self.uA, -self.uA) * self.uA

            if self.init == 1:
                q_output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
                ori_output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
                # self.beta.data.fill_(ori_output.abs().mean() / q_output.abs().mean())

            layer_idx = self.layer_num % 6

            bit_grad = torch.tensor(backward_bit)
            bit_grad = torch.tensor(8.0)
            self.bit = bit_grad

            self.bit = torch.tensor(8.0)

            output = F.conv2d(Qactivation, Qweight, Qbias, self.stride, self.padding, self.dilation, self.groups)

            self.quant_output_buff = output
            if self.training:
                self.quant_output_buff.retain_grad()
            
            moment = 1

            if bit_grad <= 16:
                output = quantize_grad(output, bit_grad, self.init, self.clip_ratio, moment)

            self.output_buff = output
            if self.training:
                self.output_buff.retain_grad()

            if self.init == 1:
                self.init.data.fill_(torch.tensor(0))

            # output = torch.abs(self.beta) * output

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

        self.quant_output_buff = None
        self.output_buff = None

        self.moment_ratio = torch.tensor(0.0)
        self.bit = torch.tensor(1.0)
        self.alpha = torch.tensor(1.0)
        # self.clip_ratio = torch.tensor(1.0)
        self.max_ratio = torch.tensor(0.0)
        self.max_std_ratio = torch.tensor(0.0)
        self.max_ratio_constant = torch.tensor(0.0)
        self.std_mean_ratio = torch.tensor(0.0)
        self.var_mean_ratio = torch.tensor(0.0)
        self.error_ratio = torch.tensor(0.0)

        self.std_ratio = torch.tensor(0.0)
        self.mean_ratio = torch.tensor(0.0)

        if self.is_quan:
            # using int32 max/min as init and backprogation to optimization
            # Weight
            self.sW = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
            self.register_buffer('init', torch.tensor(1).float())
            self.register_buffer('clip_ratio', torch.tensor(1.0).float())
            self.register_buffer('grad_max', torch.tensor(-1.0).float())
            # self.beta = nn.Parameter(data = torch.tensor(0.2).float())

            # Activation input
            if self.quan_input:
                self.uA = nn.Parameter(data = torch.tensor(2 ** 31 - 1).float())
                # self.lA = nn.Parameter(data = torch.tensor(0).float())

            if self.bias_T:
                self.sB = nn.Parameter(data = torch.tensor(2 ** 31 - 1).float())
                # self.betaB = nn.Parameter(data = torch.tensor(2 ** 31 - 1).float())

    def w_quan(self, x, s):
        global forward_bit

        bit = forward_bit
        bit = 8
        bit_range = (2 ** bit) - 2

        out = (x / torch.abs(s))
        out = (out + 1) * 0.5
        
        out = torch.clamp(out, min=0, max=1)
        out = RoundFunction.apply(out, bit_range)

        out = (2 * out) - 1

        return out

    def a_quan(self, x, u, l):
        delta = (u - l)

        out = (x - l) / delta

        global forward_bit
        bit = forward_bit
        bit = 8
        bit_range = (2 ** bit) - 1

        out = torch.clamp(out, min=0, max=1)
        out = RoundFunction.apply(out, bit_range)

        return out

    def forward(self, x):

        if self.is_quan:
            global backward_bit

            if self.init == 1:
                # print(self.init)
                # self.sW.data = self.weight.std() * 3
                self.sW.data.fill_(self.weight.std() * 3)
                self.uA.data = (x.std() / math.sqrt(1 - 2/math.pi)) * 3
                # self.uA.data.fill_(x.std() * 3)

                if self.bias_T:
                    self.sB.data.fill_(self.bias.std() * 3)
                    # self.betaB.data.fill_(self.bias.std() * 3)

            if self.quan_input:
                curr_running_la = 0
                curr_running_ua = self.uA

            Qweight = self.w_quan(self.weight, self.sW) * self.sW

            if self.bias_T:
                Qbias = self.w_quan(self.bias, self.sB) * self.sB
            else:
                Qbias = self.bias

            # Input(Activation)
            Qactivation = x

            if self.quan_input:
                Qactivation = self.a_quan(Qactivation, curr_running_ua, curr_running_la) * self.uA

            if self.init == 1:
                q_output = F.linear(Qactivation, Qweight, Qbias)
                ori_output = F.linear(x, self.weight, self.bias)

                # self.beta.data = torch.mean(torch.abs(ori_output)) / \
                #                  torch.mean(torch.abs(q_output))

            if self.init == 1:
                self.init.data.fill_(torch.tensor(0))

            output = F.linear(Qactivation, Qweight, Qbias)

            bit_grad = torch.tensor(backward_bit)
            bit_grad = torch.tensor(8.0)
            self.bit = bit_grad

            self.quant_output_buff = output
            if self.training:
                self.quant_output_buff.retain_grad()
            
            moment = 1

            if bit_grad <= 16:
                output = quantize_grad(output, bit_grad, self.init, self.clip_ratio, moment)

            self.output_buff = output
            if self.training:
                self.output_buff.retain_grad()

            # output = torch.abs(self.beta) * output

        else:
            output =  F.linear(x, self.weight, self.bias)

        return output