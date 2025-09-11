##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##############################################################################
# Random Search and Reproducibility for Neural Architecture Search, UAI 2019 # 
##############################################################################
import json
import torch, random
import torch.nn as nn
from copy import deepcopy
from models.cell_operations import ResNetBasicblock
from models.search_cells    import NAS201SearchCell as SearchCell
from utils.genotypes        import Structure


class UniformRandomSupernet(nn.Module):
  def __init__(self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats):
    super(UniformRandomSupernet, self).__init__()
    self._C        = C
    self._layerN   = N
    self.max_nodes = max_nodes
    self.stem = nn.Sequential(
                    nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                    #nn.BatchNorm2d(C))
                    nn.BatchNorm2d(C, affine=affine, track_running_stats=track_running_stats))

    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    C_prev, num_edge, edge2index = C, None, None
    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      if reduction:
        #cell = ResNetBasicblock(C_prev, C_curr, 2)
        cell = ResNetBasicblock(C_prev, C_curr, 2, affine, track_running_stats)
      else:
        cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats)
        if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
        else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
      self.cells.append(cell)
      C_prev = cell.out_dim
    self.op_names   = deepcopy(search_space)
    self._Layer     = len(self.cells)
    #self.lastact    = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    self.lastact    = nn.Sequential(
        nn.BatchNorm2d(C_prev, affine=affine, track_running_stats=track_running_stats),
        nn.ReLU(inplace=True)
    )
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    with open(f"./eff_num_of_nonlinearity_0.txt", "r") as fp: self.partition0 = json.load(fp)
    with open(f"./eff_num_of_nonlinearity_1.txt", "r") as fp: self.partition1 = json.load(fp)
    with open(f"./eff_num_of_nonlinearity_2.txt", "r") as fp: self.partition2 = json.load(fp)
    with open(f"./eff_num_of_nonlinearity_3.txt", "r") as fp: self.partition3 = json.load(fp)
    self.partition01 = self.partition0 + self.partition1
    print(len(self.partition01), "=", len(self.partition0), "+", len(self.partition1))
    print(len(self.partition2))
    print(len(self.partition3))


  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

#  def random_genotype(self):
#    genotypes = []
#    for i in range(1, self.max_nodes):
#      xlist = []
#      for j in range(i):
#        op_name = random.choice(self.op_names)
#        xlist.append((op_name, j))
#      genotypes.append(tuple(xlist))
#    arch = Structure(genotypes)
#    return arch

  ## NOTE: This is to avoid Trash Archs
  def random_genotype(self):
      genotypes = []
      for i in range(1, self.max_nodes):
          xlist = []
          if i==self.max_nodes-1:
              if genotypes[0][0][0]=='none':
                  if genotypes[1][0][0]=='none':
                      for j in range(i):
                          op_name = random.choice(self.op_names[1:])
                          xlist.append((op_name, j))
                  else:
                      for j in range(i):
                          if j==i-1 and xlist[0][0]=='none':
                              op_name = random.choice(self.op_names[1:])
                              xlist.append((op_name, j))
                          else:
                              op_name = random.choice(self.op_names)
                              xlist.append((op_name, j))
              else:
                  for j in range(i):
                      if j==i-1 and xlist[0][0]=='none' and xlist[1][0]=='none':
                          op_name = random.choice(self.op_names[1:])
                          xlist.append((op_name, j))
                      else:
                          op_name = random.choice(self.op_names)
                          xlist.append((op_name, j))
          else:
              for j in range(i):
                  op_name = random.choice(self.op_names)
                  xlist.append((op_name, j))
          genotypes.append(tuple(xlist))
      arch = Structure(genotypes)
      return arch

  def check_valid(self, arch):
      op_names = {}
      op_names['1<-0'] = arch[0][0][0]
      op_names['2<-0'] = arch[1][0][0]
      op_names['2<-1'] = arch[1][1][0]
      op_names['3<-0'] = arch[2][0][0]
      op_names['3<-1'] = arch[2][1][0]
      op_names['3<-2'] = arch[2][2][0]

      case1 = op_names['1<-0']=='none' and op_names['2<-0']=='none' and op_names['3<-0']=='none'
      case2 = op_names['1<-0']=='none' and op_names['3<-0']=='none' and op_names['3<-2']=='none'
      case3 = op_names['3<-0']=='none' and op_names['3<-1']=='none' and op_names['3<-2']=='none'

      if any([case1,case2,case3]):
          return 0
      else:
          return 1

  def random_genotype_wotrash_balanced(self):
      tind = torch.randint(0,3,(1,))
      if tind==0:
          selected = self.partition01
      elif tind==1:
          selected = self.partition2
      else:
          selected = self.partition3

      genotypes = selected[torch.randint(0,len(selected),(1,))]
      while not self.check_valid(genotypes):
          genotypes = selected[torch.randint(0,len(selected),(1,))]

      arch = Structure(genotypes)
      return arch


  def forward(self, inputs, arch=None):
    feature = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      if isinstance(cell, SearchCell):
        #if self.training:
        #  feature = cell.forward_dynamic(feature, self.random_genotype())
        #else:
        if arch is not None:
          feature = cell.forward_dynamic(feature, arch)
        else:
          feature = cell.forward_dynamic(feature, self.random_genotype())
      else:
        feature = cell(feature)

    out = self.lastact(feature)
    out = self.global_pooling(out)
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)
    return out, logits
