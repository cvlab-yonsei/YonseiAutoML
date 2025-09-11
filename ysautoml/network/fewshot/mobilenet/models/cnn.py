import math
import torch.nn as nn
from ..models.layers import OPS


class CNN(nn.Module):
    def __init__(self, search_space, arch, drop_out, n_class=1000, input_size=224, width_mult=1.):
        super(CNN, self).__init__()

        # self.interverted_residual_setting = [
        #     # channel, layers, stride
        #     [32,  4, 2],
        #     [56,  4, 2],
        #     [112, 4, 2],
        #     [128, 4, 1],
        #     [256, 4, 2],
        #     [432, 1, 1],
        # ]
        # input_channel    = int(40 * width_mult)
        # first_cell_width = int(24 * width_mult)

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

        affine, track_running_stats = True, True
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel, affine=affine, track_running_stats=track_running_stats),
            nn.ReLU6(inplace=True),
        )

        self.first_block = OPS['3x3_MBConv1'](input_channel, first_cell_width, 1, affine, track_running_stats)
        input_channel = first_cell_width

        self.blocks = nn.ModuleList()
        ind = 0
        for c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                op_ind  = arch[ind]
                op_name = search_space[op_ind]
                ind += 1
                if i == 0:
                    tmp = OPS[op_name](input_channel, output_channel, s, affine, track_running_stats)
                else:
                    tmp = OPS[op_name](input_channel, output_channel, 1, affine, track_running_stats)
                self.blocks.append( tmp )
                input_channel = output_channel

        # last_channel = int(1728 * width_mult)
        last_channel = int(1280 * width_mult)
        self.feature_mix_layer = nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel, affine=affine, track_running_stats=track_running_stats),
            nn.ReLU6(inplace=True)
        ) 
        self.avgpool    = nn.AvgPool2d(input_size//32)
        self.classifier = nn.Sequential(
            nn.Dropout(drop_out),
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

    def forward(self, x):
        x = self.first_conv(x)
        x = self.first_block(x)
        for ops in self.blocks:
            x = ops(x)
        x = self.feature_mix_layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
