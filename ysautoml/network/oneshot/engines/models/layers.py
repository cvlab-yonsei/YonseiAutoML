import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['OPS', 'SearchSpaceNames']


OPS = {
    '3x3_MBConv1': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 3, 1, stride, 1, affine, track_running_stats),

    '3x3_MBConv3': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 3, 1, stride, 3, affine, track_running_stats),
    '3x3_MBConv6': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 3, 1, stride, 6, affine, track_running_stats),
    '5x5_MBConv3': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 5, 2, stride, 3, affine, track_running_stats),
    '5x5_MBConv6': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 5, 2, stride, 6, affine, track_running_stats),
    '7x7_MBConv3': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 7, 3, stride, 3, affine, track_running_stats),
    '7x7_MBConv6': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(C_in, C_out, 7, 3, stride, 6, affine, track_running_stats),

    'Identity'   : lambda C_in, C_out, stride, affine, track_running_stats: Identity(),

    '3x3_MBConv3_SE': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual_SE(C_in, C_out, 3, 1, stride, 3, affine, track_running_stats),
    '3x3_MBConv6_SE': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual_SE(C_in, C_out, 3, 1, stride, 6, affine, track_running_stats),
    '5x5_MBConv3_SE': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual_SE(C_in, C_out, 5, 2, stride, 3, affine, track_running_stats),
    '5x5_MBConv6_SE': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual_SE(C_in, C_out, 5, 2, stride, 6, affine, track_running_stats),
    '7x7_MBConv3_SE': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual_SE(C_in, C_out, 7, 3, stride, 3, affine, track_running_stats),
    '7x7_MBConv6_SE': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual_SE(C_in, C_out, 7, 3, stride, 6, affine, track_running_stats),

}


PROXYLESS_SPACE = [
    '3x3_MBConv3', '3x3_MBConv6', 
    '5x5_MBConv3', '5x5_MBConv6', 
    '7x7_MBConv3', '7x7_MBConv6',
    'Identity'
]


GREEDY_SPACE = [
    '3x3_MBConv3', '3x3_MBConv6', 
    '5x5_MBConv3', '5x5_MBConv6', 
    '7x7_MBConv3', '7x7_MBConv6',
    '3x3_MBConv3_SE', '3x3_MBConv6_SE', 
    '5x5_MBConv3_SE', '5x5_MBConv6_SE', 
    '7x7_MBConv3_SE', '7x7_MBConv6_SE',
    'Identity'
]



SearchSpaceNames = {
    "proxyless": PROXYLESS_SPACE, 
    "greedy": GREEDY_SPACE,
}


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


class InvertedResidual_SE(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, padding, stride, expand_ratio, affine, track_running_stats):
        super(InvertedResidual_SE, self).__init__()
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

        self.se = SqueezeExcite(hidden_dim, rd_ratio=1/expand_ratio)

    def forward(self, x):
        inputs = x

        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.se(x)
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


class SqueezeExcite(nn.Module):
    """ Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family
    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
            self, in_chs, rd_ratio=0.25, rd_channels=None, act_layer=nn.ReLU,
            gate_layer=nn.Sigmoid, force_act_layer=None, rd_round_fn=None):
        super(SqueezeExcite, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = nn.ReLU6(inplace=True) #create_act_layer(act_layer, inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = nn.Sigmoid() #create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)
