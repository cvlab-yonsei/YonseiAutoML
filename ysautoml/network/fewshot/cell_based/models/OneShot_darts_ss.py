##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##############################################################################
# Random Search and Reproducibility for Neural Architecture Search, UAI 2019 # 
##############################################################################
import json
import torch, random
import torch.nn as nn
from copy import deepcopy
from models.search_cells    import NASNetSearchCell as SearchCell
from utils.genotypes        import Structure


class UniformRandomSupernet(nn.Module):
  def __init__(self, C, layers, num_classes, search_space, steps=4, multiplier=4, stem_multiplier=3, affine=False, track_running_stats=False):
    super(UniformRandomSupernet, self).__init__()
    self._C          = C
    self._layers     = layers
    self._steps      = steps
    self._multiplier = multiplier
    self.op_names    = deepcopy(search_space)

    C_curr = stem_multiplier * C
    self.stem = nn.Sequential(
                    nn.Conv2d(3, C_curr, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(C_curr, affine=affine, track_running_stats=track_running_stats))


    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = SearchCell(search_space, steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, affine, track_running_stats)
      self.cells += [cell]
      reduction_prev = reduction
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, L={_layers}, Step={_steps}, Multiplier={_multiplier})'.format(name=self.__class__.__name__, **self.__dict__))

  def random_genotype(self):
    genotypes = []
    for i in range(self._steps):
      xlist = []
      for j in range(i+2):
        op_name = random.choice(self.op_names)
        xlist.append((op_name, j))
      genotypes.append(tuple(xlist))
    arch = Structure(genotypes)
    return arch

  def forward(self, inputs, arch=None):
    if arch is None:
      arch = self.random_genotype()

    s0 = s1 = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell.forward_dynamic(s0, s1, arch)

    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits
