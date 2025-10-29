import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting


class IMTL_L(AbsWeighting):
    r"""Impartial Multi-task Learning (IMTL).
    
    This method is proposed in `Towards Impartial Multi-task Learning (ICLR 2021) <https://openreview.net/forum?id=IMPnRXEWpvr>`_ \
    and implemented by us.

    """
    def __init__(self):
        super(IMTL_L, self).__init__()
    
    def init_param(self):
        self.loss_scale = nn.Parameter(torch.tensor([0.0]*self.task_num, device=self.device))
        
    def backward(self, losses, **kwargs):
        losses = self.loss_scale.exp()*losses - self.loss_scale
        loss = torch.sum(losses)
        loss.backward()
        return self.loss_scale.exp().detach().cpu().numpy()
