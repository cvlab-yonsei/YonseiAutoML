import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['OPS', 'PROXYLESSNAS_SPACE']


OPS = {
    '3x3_MBConv1': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 3, 1, stride, 1, affine, track_running_stats),
    '3x3_MBConv3': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 3, 1, stride, 3, affine, track_running_stats),
    '3x3_MBConv6': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 3, 1, stride, 6, affine, track_running_stats),
    '5x5_MBConv3': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 5, 2, stride, 3, affine, track_running_stats),
    '5x5_MBConv6': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 5, 2, stride, 6, affine, track_running_stats),
    '7x7_MBConv3': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 7, 3, stride, 3, affine, track_running_stats),
    '7x7_MBConv6': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 7, 3, stride, 6, affine, track_running_stats),
    'Identity'   : lambda C_in, C_out, stride, affine, track_running_stats: Identity(),
}


PROXYLESSNAS_SPACE = [
    '3x3_MBConv3', '3x3_MBConv6', 
    '5x5_MBConv3', '5x5_MBConv6', 
    '7x7_MBConv3', '7x7_MBConv6',
    'Identity'
]


class InvertedResidual(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, padding, stride, expand_ratio, affine, track_running_stats):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio

        self.use_res_connect = self.stride == 1 and C_in == C_out

        hidden_dim = round(C_in * expand_ratio)

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(
                nn.Conv2d(C_in, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim, affine=affine, track_running_stats=track_running_stats),
                nn.ReLU6(inplace=True),
            )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim, affine=affine, track_running_stats=track_running_stats),
            nn.ReLU6(inplace=True),
        )

        self.point_linear = nn.Sequential(
            nn.Conv2d(hidden_dim, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
        )

    def forward(self, x):
        inputs = x

        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x) 
        x = self.depth_conv(x)
        x = self.point_linear(x)

        if self.use_res_connect:
            return inputs + x
        else:
            return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

