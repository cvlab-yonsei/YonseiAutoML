import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class SI(AbsWeighting):
    r"""Scale Invariant (SI).

    """
    def __init__(self):
        super(SI, self).__init__()
        
    def backward(self, losses, **kwargs):
        loss = torch.log(losses).sum()
        loss.backward()
        return np.ones(self.task_num)