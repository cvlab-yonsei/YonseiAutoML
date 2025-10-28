import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LinearSub(nn.Linear):
    def __init__(self, in_dim, out_dim, super_out_dim=None, bias=True, uniform_=None, non_linear='linear', scale=False):
        super().__init__(in_dim, out_dim, bias=bias)

        self.sample_in_dim = in_dim
        self.sample_out_dim = out_dim
        if super_out_dim is None:
            super_out_dim = out_dim
        self.sample_scale = super_out_dim/self.sample_out_dim

        self.scale = scale
        self._reset_parameters(bias, uniform_, non_linear)

        self.samples = {}
        self.samples['weight'] = self.weight
        self.samples['bias'] = self.bias

    def profile(self, mode=True):
        self.profiling = mode

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        return


    def forward(self, x):
        return F.linear(x, self.weight, self.bias) * (self.sample_scale if self.scale else 1)

    def calc_sampled_param_num(self):
        weight_numel = self.weight.numel()

        if self.bias is not None:
            bias_numel = self.bias.numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length *  np.prod(self.weight.size())
        return total_flops

