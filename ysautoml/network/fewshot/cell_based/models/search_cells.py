##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import math, random, torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from models.cell_operations import OPS


# This module is used for NAS-Bench-201, represents a small search space with a complete DAG
class NAS201SearchCell(nn.Module):
  def __init__(self, C_in, C_out, stride, max_nodes, op_names, affine=False, track_running_stats=True, pcdarts_prune_ratio=1):
    super(NAS201SearchCell, self).__init__()

    self.op_names  = deepcopy(op_names)
    self.edges     = nn.ModuleDict()
    self.max_nodes = max_nodes
    self.in_dim    = C_in
    self.out_dim   = C_out
    for i in range(1, max_nodes):
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        if j == 0:
          xlists = [OPS[op_name](C_in // pcdarts_prune_ratio, C_out // pcdarts_prune_ratio, stride, affine, track_running_stats) for op_name in op_names]
        else:
          xlists = [OPS[op_name](C_in // pcdarts_prune_ratio, C_out // pcdarts_prune_ratio,      1, affine, track_running_stats) for op_name in op_names]
        self.edges[ node_str ] = nn.ModuleList( xlists )
    self.edge_keys  = sorted(list(self.edges.keys()))
    self.edge2index = {key:i for i, key in enumerate(self.edge_keys)}
    self.num_edges  = len(self.edges)

  def extra_repr(self):
    string = 'info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}'.format(**self.__dict__)
    return string

  def channel_shuffle(self, x, groups):
    """
    https://github.com/yuhuixu1993/PC-DARTS/blob/86446d1b6bbbd5f752cc60396be13d2d5737a081/model_search.py#L9
    """

    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

  def forward(self, inputs, weightss):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = weightss[ self.edge2index[node_str] ]
        inter_nodes.append( sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # PCDARTs
  def forward_pcdarts(self, inputs, weightss):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        dim_2 = nodes[j].shape[1]
        xtemp = nodes[j][:, :  dim_2 // 4, :, :]
        xtemp2 = nodes[j][:, dim_2 // 4:, :, :]
        node_str = '{:}<-{:}'.format(i, j)
        weights = weightss[self.edge2index[node_str]]
        temp1 = sum(layer(xtemp) * w for layer, w in zip(self.edges[node_str], weights))
        ans = torch.cat([temp1, xtemp2], dim=1)
        ans = self.channel_shuffle(ans, 4)
        inter_nodes.append(ans)
      nodes.append(sum(inter_nodes))
    return nodes[-1]

  # GDAS
  def forward_gdas(self, inputs, hardwts, index):
    nodes   = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = hardwts[ self.edge2index[node_str] ]
        argmaxs  = index[ self.edge2index[node_str] ].item()
        weigsum  = sum( weights[_ie] * edge(nodes[j]) if _ie == argmaxs else weights[_ie] for _ie, edge in enumerate(self.edges[node_str]) )
        inter_nodes.append( weigsum )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # joint
  def forward_joint(self, inputs, weightss):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = weightss[ self.edge2index[node_str] ]
        #aggregation = sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) / weights.numel()
        aggregation = sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) )
        inter_nodes.append( aggregation )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # uniform random sampling per iteration, SETN
  def forward_urs(self, inputs):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      while True: # to avoid select zero for all ops
        sops, has_non_zero = [], False
        for j in range(i):
          node_str   = '{:}<-{:}'.format(i, j)
          candidates = self.edges[node_str]
          select_op  = random.choice(candidates)
          sops.append( select_op )
          if not hasattr(select_op, 'is_zero') or select_op.is_zero is False: has_non_zero=True
        if has_non_zero: break
      inter_nodes = []
      for j, select_op in enumerate(sops):
        inter_nodes.append( select_op(nodes[j]) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # select the argmax
  def forward_select(self, inputs, weightss):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = weightss[ self.edge2index[node_str] ]
        inter_nodes.append( self.edges[node_str][ weights.argmax().item() ]( nodes[j] ) )
        #inter_nodes.append( sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # forward with a specific structure
  def forward_dynamic(self, inputs, structure):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      cur_op_node = structure.nodes[i-1]
      inter_nodes = []
      for op_name, j in cur_op_node:
        node_str = '{:}<-{:}'.format(i, j)
        op_index = self.op_names.index( op_name )
        inter_nodes.append( self.edges[node_str][op_index]( nodes[j] ) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]


class MixedOp(nn.Module):
  def __init__(self, space, C, stride, affine, track_running_stats):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in space:
      op = OPS[primitive](C, C, stride, affine, track_running_stats)
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))
       

# Learning Transferable Architectures for Scalable Image Recognition, CVPR 2018
class NASNetSearchCell(nn.Module):
  def __init__(self, space, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, affine, track_running_stats):
    super(NASNetSearchCell, self).__init__()
    self.reduction = reduction

    self.op_names  = deepcopy(space)
    if reduction_prev: 
      self.preprocess0 = OPS['skip_connect'](C_prev_prev, C, 2, affine, track_running_stats)
    else:
      self.preprocess0 = OPS['nor_conv_1x1'](C_prev_prev, C, 1, affine, track_running_stats)

    self.preprocess1 = OPS['nor_conv_1x1'](C_prev, C, 1, affine, track_running_stats)
    self._steps = steps
    self._multiplier = multiplier

    self.edges = nn.ModuleDict()
    for i in range(self._steps):
      for j in range(i+2):
        node_str = '{:}<-{:}'.format(i+2, j)  # indicate the edge from node-(j) to node-(i+2)
        stride = 2 if reduction and j < 2 else 1
#        op = MixedOp(space, C, stride, affine, track_running_stats)
#        self.edges[ node_str ] = op

        xlists = []
        for primitive in space:
          op = OPS[primitive](C, C, stride, affine, track_running_stats)
          if 'pool' in primitive: # NOTE: added to be the same as 'MixedOp' in https://github.com/quark0/darts/blob/master/cnn/model_search.py#L10
            op = nn.Sequential(op, nn.BatchNorm2d(C, affine=affine, track_running_stats=track_running_stats))
          xlists.append(op)
        self.edges[ node_str ] = nn.ModuleList( xlists )

    self.edge_keys  = sorted(list(self.edges.keys()))
    self.edge2index = {key:i for i, key in enumerate(self.edge_keys)}
    self.num_edges  = len(self.edges)


  # forward with a specific structure
  def forward_dynamic(self, s0, s1, structure):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    states = [s0, s1]
    for i in range(self._steps):
      cur_op_node = structure.nodes[i]
      inter_nodes = []
      for op_name, j in cur_op_node:
        node_str = '{:}<-{:}'.format(i+2, j)
        op_index = self.op_names.index( op_name )
        inter_nodes.append( self.edges[node_str][op_index]( states[j] ) )
      states.append( sum(inter_nodes) )
    return torch.cat(states[-self._multiplier:], dim=1)
