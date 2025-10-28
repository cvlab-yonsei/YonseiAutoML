import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import to_2tuple
import numpy as np

class PatchembedSub(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, super_embed_dim=768, scale=False):
        super(PatchembedSub, self).__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.scale = scale

    # sampled_
        self.sample_embed_dim = embed_dim
        self.sampled_weight = self.proj.weight
        self.sampled_bias = self.proj.bias
        self.sampled_scale = super_embed_dim / self.sample_embed_dim

    def set_sample_config(self, sample_embed_dim):
        return
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1,2)
        if self.scale:
            return x * self.sampled_scale
        return x
    def calc_sampled_param_num(self):
        return  self.sampled_weight.numel() + self.sampled_bias.numel()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias is not None:
             total_flops += self.sampled_bias.size(0)
        total_flops += sequence_length * np.prod(self.sampled_weight.size())
        return total_flops