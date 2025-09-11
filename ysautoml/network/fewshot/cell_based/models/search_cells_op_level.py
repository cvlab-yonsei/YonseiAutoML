##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import math, random, torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from models.cell_operations_op_level import OPS, ReLUConvBN


class ResNetBasicblock_splitable(nn.Module):
    def __init__(self, inplanes, planes, stride, 
                 affine=True, track_running_stats=True,
                 split_op_list=None):
        super().__init__()
        
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.split_op_list = split_op_list
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, affine, track_running_stats)
        self.conv_b = ReLUConvBN(  planes, planes, 3,      1, 1, 1, affine, track_running_stats)
        if stride == 2:
            self.downsample = nn.Sequential(
              nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
              nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
            )
        elif inplanes != planes:
            self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1, affine, track_running_stats)
        else:
            self.downsample = None
        self.in_dim  = inplanes
        self.out_dim = planes
        self.stride  = stride
        self.num_conv = 2

    def extra_repr(self):
        string = '{name}(split_op_list={split_op_list}, inC={in_dim}, outC={out_dim}, stride={stride})'.format(name=self.__class__.__name__, **self.__dict__)
        return string

    def forward(self, input_list):
        output_list = []
        for inputs in input_list:
            basicblock = self.conv_a(inputs)
            basicblock = self.conv_b(basicblock)

            if self.downsample is not None:
                residual = self.downsample(inputs)
            else:
                residual = inputs
            output_list.append( residual + basicblock )
        return output_list


class NAS201SearchCell_splitable(nn.Module):
    def __init__(self, C_in, C_out, stride, max_nodes, op_names, 
                 affine=False, track_running_stats=True,
                 split_op_list=None):
        super().__init__()

        self.op_names  = deepcopy(op_names) # ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        self.edges     = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim    = C_in
        self.out_dim   = C_out
        self.split_op_list = split_op_list
        
#         {'1<-0': ['nor_conv_1x1'],
#          '2<-0': ['nor_conv_1x1'],
#          '2<-1': ['nor_conv_1x1'],
#          '3<-0': ['nor_conv_3x3'],
#          '3<-1': ['nor_conv_1x1'],
#          '3<-2': ['nor_conv_1x1', 'nor_conv_3x3']}
        
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                if j == 0:
                    xlists = [OPS[op_name](C_in, C_out, stride, affine, track_running_stats) for op_name in op_names]
                else:
                    xlists = [OPS[op_name](C_in, C_out, 1, affine, track_running_stats) for op_name in op_names]

                if split_op_list is not None and node_str in split_op_list:
                    for op_name in op_names:
                        if op_name in split_op_list[node_str]:
                            if j==0:
                                add_op = OPS[op_name](C_in, C_out, stride, affine, track_running_stats)
                            else:
                                add_op = OPS[op_name](C_in, C_out, 1, affine, track_running_stats)
                            xlists += [add_op]
            
                self.edges[node_str] = nn.ModuleList(xlists)
        self.edge_keys  = sorted(list(self.edges.keys()))
        self.edge2index = {key:i for i, key in enumerate(self.edge_keys)}
        self.num_edges  = len(self.edges)

    def extra_repr(self):
        string = 'info :: split_op_list={split_op_list}, {max_nodes} nodes, inC={in_dim}, outC={out_dim}'.format(**self.__dict__)
        return string

    def ordered_sum(self, fir_list, sec_list):
        tot = []
        for aa in fir_list[0::2]:
            for bb in sec_list[0::2]:
                tot += [aa+bb]
        for aa in fir_list[1::2]:
            for bb in sec_list[1::2]:
                tot += [aa+bb]
        return tot
    
    def list_sum(self, fir_list, sec_list):
        tot = []
        for aa in fir_list:
            for bb in sec_list:
                tot += [aa+bb]
        return tot

    def forward_dynamic(self, input_list, structure):
        warning = False
        output_list = None
        for inputs in input_list:
            nodes = [ [inputs] ]
            for i in range(1, self.max_nodes):
                cur_op_node = structure.nodes[i-1]
                inter_nodes = [] ## [ [], [], ... , [] ]
                for op_name, j in cur_op_node:
                    node_str = '{:}<-{:}'.format(i, j)
                    op_index = self.op_names.index( op_name )
                    tmp_nodes = self.edges[node_str][op_index](nodes[j])
                    
                    if self.split_op_list is not None and node_str in self.split_op_list:
                        if op_name in self.split_op_list[node_str]:
                            if node_str == '1<-0': warning = True
                            tmp_nodes += self.edges[node_str][-1](nodes[j]) # HARD CODED
                            
                    inter_nodes.append( tmp_nodes )
####################################################
#                 tmp_n = inter_nodes[0]
#                 for ind in range(1, len(inter_nodes)):
#                     tmp_n = self.list_sum(tmp_n, inter_nodes[ind])
#                 nodes.append( tmp_n )
####################################################
                if warning and i==self.max_nodes-1: 
                    # NOTE: HARD CODED
                    tmp_n = self.list_sum(
                        inter_nodes[0], 
                        self.ordered_sum(inter_nodes[1], inter_nodes[2])
                    )
                else:
                    tmp_n = inter_nodes[0]
                    for ind in range(1, len(inter_nodes)):
                        tmp_n = self.list_sum(tmp_n, inter_nodes[ind])
                nodes.append( tmp_n )
####################################################
            if isinstance(output_list, list):
                output_list += nodes[-1]
            else:
                output_list = nodes[-1]
        return output_list
