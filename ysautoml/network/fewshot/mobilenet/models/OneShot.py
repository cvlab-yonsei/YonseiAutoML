import math
import torch.nn as nn
# from ..models.layers import OPS, Identity
# --- Flexible import fix: works both as script & module ---
try:
    # 정상적인 패키지 실행 (python -m ...)
    from ..models.layers import OPS, Identity
except (ImportError, ValueError):
    # 단독 실행 또는 API subprocess 실행 시 경로 보정
    import os, sys
    cur_dir = os.path.dirname(os.path.abspath(__file__))  # .../models
    parent_dir = os.path.dirname(cur_dir)                 # .../mobilenet
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from models.layers import OPS, Identity
# ------------------------------------------------------------



class SuperNet(nn.Module):
    def __init__(self, search_space, affine, track_running_stats, n_class=1000, input_size=224, width_mult=1.):
        super(SuperNet, self).__init__()
        
        self.interverted_residual_setting = [
            # channel, layers, stride
            #[24,  4, 2],
            [32,  4, 2],
            [40,  4, 2],
            [80,  4, 2],
            [96,  4, 1],
            [192, 4, 2],
            [320, 1, 1],
        ]
        input_channel    = int(32 * width_mult)
        first_cell_width = int(16 * width_mult)

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel, affine=affine, track_running_stats=track_running_stats),
            nn.ReLU6(inplace=True),
        )

        self.first_block = OPS['3x3_MBConv1'](input_channel, first_cell_width, 1, affine, track_running_stats)
        input_channel = first_cell_width

        self.blocks  = nn.ModuleList()
        self.choices = []
        for c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    op_list = [OPS[op_name](input_channel, output_channel, s, affine, track_running_stats) for op_name in search_space[:-1]]
                else:
                    op_list = [OPS[op_name](input_channel, output_channel, 1, affine, track_running_stats) for op_name in search_space]
                op_list = nn.ModuleList(op_list)
                self.blocks.append( op_list )
                self.choices.append( len(op_list) )
                input_channel = output_channel

        last_channel = int(1280 * width_mult)
        self.feature_mix_layer = nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel, affine=affine, track_running_stats=track_running_stats),
            nn.ReLU6(inplace=True)
        ) 
        self.avgpool    = nn.AvgPool2d(input_size//32)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, n_class),
        )

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
            if hasattr(ops[op_ind], 'se'):
                flop_se1, out = count_conv_flop(ops[op_ind].se.conv_reduce, x.mean((2, 3), keepdim=True))
                flop_se2, out = count_conv_flop(ops[op_ind].se.conv_expand, out)
                flop_se3 = ops[op_ind].se.conv_expand.out_channels*x.shape[2]*x.shape[3]
                flops += flop_se1 + flop_se2 + flop_se3
            flops += count_conv_flop(ops[op_ind].point_linear[0], ss[2])

        flops += count_conv_flop(self.feature_mix_layer[0], sizes[-1][-1])
        flops += self.classifier[0].weight.numel()
        return flops

    def forward(self, x, arch):
        x = self.first_conv(x)
        x = self.first_block(x)

        for ops, op_ind in zip(self.blocks, arch):
            x = ops[op_ind](x)
        x = self.feature_mix_layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
