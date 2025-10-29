import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class SI_naive(AbsWeighting):
    r"""Scale Invariant (SI) - naive version.

    """
    def __init__(self):
        super(SI_naive, self).__init__()
        
    def backward(self, losses, **kwargs):
        loss = torch.mul(losses, (1/losses.detach())).sum()
        loss.backward()
        return np.ones(self.task_num)