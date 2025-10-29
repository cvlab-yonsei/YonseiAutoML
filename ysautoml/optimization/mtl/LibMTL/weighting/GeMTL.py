import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class GeMTL(AbsWeighting):
    r"""Achievement-based Multi-task Learning (AMTL).
    
    This method is proposed in `Achievement-based Training Progress Balancing for Multi-Task Learning (ICCV 2023) <https://openaccess.thecvf.com/content/ICCV2023/html/Yun_Achievement-Based_Training_Progress_Balancing_for_Multi-Task_Learning_ICCV_2023_paper.html>`_ \
    and implemented by us.

    """
    def __init__(self):
        super(GeMTL, self).__init__()
        
    def backward(self, losses, **kwargs):
        # Load hyperparameters
        if not hasattr(self, 'potentials'):
            self.p = -0.5
            
        self.p += 1 / (self.train_batch * self.epochs) 
        
        if abs(self.p)<0.1:
            loss = torch.pow( losses.prod(), 1./self.task_num) # GM
        else:
            loss = torch.pow( torch.pow(losses, self.p).sum() / self.task_num, 1/self.p) # GeM

        loss.backward()
        batch_weight = losses / (self.task_num * losses.prod())
        return batch_weight.detach().cpu().numpy()