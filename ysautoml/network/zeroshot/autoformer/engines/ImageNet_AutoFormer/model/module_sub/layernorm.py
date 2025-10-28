import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormSub(torch.nn.LayerNorm):
    def __init__(self, embed_dim):
        super().__init__(embed_dim)

        self.sample_embed_dim = embed_dim

    def set_sample_config(self, sample_embed_dim):
        return

    def forward(self, x):
        return F.layer_norm(x, (self.sample_embed_dim,), weight=self.weight, bias=self.bias, eps=self.eps)

    def calc_sampled_param_num(self):
        return self.weight.numel() + self.bias.numel()

    def get_complexity(self, sequence_length):
        return sequence_length * self.sample_embed_dim
