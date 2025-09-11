import math
import torch.nn as nn
from ..models.layers import OPS, Identity


class SuperNet_decom(nn.Module):
    def __init__(self, K, thresholds, search_space, affine, track_running_stats, n_class=1000, input_size=224, width_mult=1.):
        super(SuperNet_decom, self).__init__()
        assert K-1==len(thresholds)

        KK=2 #K
        self.interverted_residual_setting = [
            # channel, layers, stride
            [32//KK,  4, 2],
            [40//KK,  4, 2],
            [80//KK,  4, 2],
            [96//KK,  4, 1],
            [192//KK, 4, 2],
            [320//KK, 1, 1],
        ]
        input_channel    = int((32//KK) * width_mult)
        first_cell_width = int((16//KK) * width_mult)

#        self.interverted_residual_setting = [
#            # channel, layers, stride
#            [32,  4, 2],
#            [40,  4, 2],
#            [80,  4, 2],
#            [96,  4, 1],
#            [192, 4, 2],
#            [320, 1, 1],
#        ]
#        input_channel    = int(32 * width_mult)
#        first_cell_width = int(16 * width_mult)

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
                    op_list = [ nn.ModuleList([OPS[op_name](input_channel, output_channel, s, affine, track_running_stats) for op_name in search_space[:-1]]) for _ in range(K) ]
                else:
                    op_list = [ nn.ModuleList([OPS[op_name](input_channel, output_channel, 1, affine, track_running_stats) for op_name in search_space]) for _ in range(K) ]
                op_list = nn.ModuleList(op_list)
                self.blocks.append( op_list )
                self.choices.append( len(op_list[0]) )
                input_channel = output_channel

        last_channel = int((1280//KK) * width_mult)
        #last_channel = int(1280 * width_mult)
        self.feature_mix_layer = nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel, affine=affine, track_running_stats=track_running_stats),
            nn.ReLU6(inplace=True)
        ) 
        self.avgpool    = nn.AvgPool2d(input_size//32)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, n_class),
        )

        self.thresholds = thresholds
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

    def forward(self, x, arch):
        x = self.first_conv(x)
        x = self.first_block(x)

        ENN = 2 * (21 - sum(arch==6)) # NOTE: HARD CODE [6, arch: tensor] only
        #if ENN <= self.thresholds[0]:
        #if ENN in [38, 40]:
#        if int(ENN) % 4 == 0:
        if int(ENN) in [36, 38]:
            k_ind = 0
        else:
            k_ind = 1 

#        if ENN == self.thresholds[0]:
#            k_ind = 0
#        elif ENN == self.thresholds[1]:
#            k_ind = 1
#        elif ENN == self.thresholds[2]:
#            k_ind = 2
#        else:
#            k_ind = 3 

#        if ENN == self.thresholds[0]:
#            k_ind = 0
#        elif ENN == self.thresholds[1]:
#            k_ind = 1
#        elif ENN == self.thresholds[2]:
#            k_ind = 2
#        elif ENN == self.thresholds[3]:
#            k_ind = 3
#        elif ENN == self.thresholds[4]:
#            k_ind = 4
#        else:
#            k_ind = 5

        for ops, op_ind in zip(self.blocks, arch):
            x = ops[k_ind][op_ind](x)
#        _accumulated = [0] 
#        for ops, op_ind in zip(self.blocks, arch):
#            if _accumulated[-1] <= self.thresholds[0]:
#                tmp = ops[0]
#            else:
#                tmp = ops[1]
#            x = tmp[op_ind](x)
#            if isinstance(tmp[op_ind], Identity):
#                _accumulated += [_accumulated[-1]]
#            else:
#                _accumulated += [_accumulated[-1] + 2]
        x = self.feature_mix_layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
