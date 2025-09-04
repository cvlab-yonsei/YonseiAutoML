import math
import torch
import torch.nn as nn
from .modeling_layers import OPS


UNIQUE_ENN = 16 # For the ProxylessNAS space: from 12 to 42


class SuperNet(nn.Module):
    def __init__(self, num_K, num_G, search_space, affine, track_running_stats, n_class=1000, input_size=224, width_mult=1.):
        super().__init__()

        self.interverted_residual_setting = [
            # channel, layers, stride
            [24//num_G,  4, 2],
            [40//num_G,  4, 2],
            [80//num_G,  4, 2],
            [96//num_G,  4, 1],
            [192//num_G, 4, 2],
            [320//num_G, 1, 1],
        ]
        input_channel    = int((32//num_G) * width_mult)
        first_cell_width = int((16//num_G) * width_mult)

        first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel, affine=affine, track_running_stats=track_running_stats),
            nn.ReLU6(inplace=True),
        )
        self.first_conv = nn.ModuleList( [first_conv for _ in range(num_K)] )

        first_block = OPS['3x3_MBConv1'](input_channel, first_cell_width, 1, affine, track_running_stats)
        self.first_block = nn.ModuleList( [first_block for _ in range(num_K)] )
        input_channel = first_cell_width

        self.search_space = search_space
        self.blocks = nn.ModuleList()
        self.choices = []
        for c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    tmp = [ OPS[op_name](input_channel, output_channel, s, affine, track_running_stats) for op_name in search_space[:-1] ]
                else:
                    tmp = [ OPS[op_name](input_channel, output_channel, 1, affine, track_running_stats) for op_name in search_space ]
                op_list = nn.ModuleList( [nn.ModuleList(tmp) for _ in range(num_K)] )
                self.blocks.append( op_list )
                self.choices.append( len(op_list[0]) )
                input_channel = output_channel

        last_channel = int((1280//num_G) * width_mult)
        feature_mix_layer = nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel, affine=affine, track_running_stats=track_running_stats),
            nn.ReLU6(inplace=True)
        ) 
        self.feature_mix_layer = nn.ModuleList( [feature_mix_layer for _ in range(num_K)] )
        self.avgpool    = nn.AvgPool2d(input_size//32)
        classifier = nn.Linear(last_channel, n_class)
        self.classifier = nn.ModuleList( [classifier for _ in range(num_K)] )

        self.initialize()

    def initialize(self): 
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # he_fout
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # if init_div_groups:
                #     n /= m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()

    def get_flops(self, arch):
        def count_conv_flop(layer, x):
            out_h = int(x / layer.stride[0])
            out_w = int(x / layer.stride[1])
            delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * out_h * out_w / layer.groups
            return delta_ops
    
        flops = count_conv_flop(self.first_conv[0], 224)
        flops += count_conv_flop(self.first_block.depth_conv[0], 112)
        flops += count_conv_flop(self.first_block.point_linear[0], 112)

        sizes = [(112, 112, 56),
                 (56,   56, 56),
                 (56,   56, 56),
                 (56,   56, 56),

                 (56,   56, 28),
                 (28,   28, 28),
                 (28,   28, 28),
                 (28,   28, 28),

                 (28,   28, 14),
                 (14,   14, 14),
                 (14,   14, 14),
                 (14,   14, 14),

                 (14,   14, 14),
                 (14,   14, 14),
                 (14,   14, 14),
                 (14,   14, 14),

                 (14,   14,  7),
                 (7,     7,  7),
                 (7,     7,  7),
                 (7,     7,  7),

                 (7,     7,  7)]
        for ops, op_ind, ss in zip(self.blocks, arch, sizes):
            if isinstance(ops[op_ind], Identity):
                continue
            flops += count_conv_flop(ops[op_ind].inverted_bottleneck[0], ss[0])
            flops += count_conv_flop(ops[op_ind].depth_conv[0], ss[1])
            flops += count_conv_flop(ops[op_ind].point_linear[0], ss[2])

        flops += count_conv_flop(self.feature_mix_layer[0], sizes[-1][-1])
        flops += self.classifier[0].weight.numel()
        return flops

    def compute_valid_enn(self, arch):
        ID_IND = self.search_space.index('Identity')
        if isinstance(arch, list):
            return 2 * (len(self.choices) - arch.count(ID_IND))
        elif isinstance(arch, torch.Tensor):
            return 2 * (len(self.choices) - sum(arch==ID_IND))
        else:
            raise TypeError("arch should be list or tensor")

    def compute_cell_id(self, valid_enn):
        if self.num_K == 1:
            cell_id = 0
        else:
            split_id = UNIQUE_ENN - self.num_K
            if valid_enn <= split_id:
                cell_id = 0
            else:
                cell_id = valid_enn - split_id
        return cell_id

    def forward(self, x, arch):
        cell_id = self.compute_cell_id(self.compute_valid_enn(arch))
        x = self.first_conv[cell_id](x)
        x = self.first_block[cell_id](x)
        for ops, op_ind in zip(self.blocks, arch):
            x = ops[cell_id][op_ind](x)
        x = self.feature_mix_layer[cell_id](x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier[cell_id](x)
        return x
