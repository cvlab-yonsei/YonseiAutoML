from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.optim as optim
import torch
from torch.optim.optimizer import Optimizer, required
import math
from torch import Tensor
from typing import List, Optional


def adam(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0.0001,
         amsgrad=False, **_):
  if isinstance(betas, str):
    betas = eval(betas)
  return optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay,
                    amsgrad=amsgrad)


def sgd(parameters, lr=0.1, momentum=0.9, weight_decay=0.0001, **_):
  return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)


def get_optimizer(config, parameters):
  f = globals().get(config.optimizer.name)
  return f(parameters, **config.optimizer.params)

def get_q_optimizer(config, parameters):
  f = globals().get(config.q_optimizer.name)
  return f(parameters, **config.q_optimizer.params)
