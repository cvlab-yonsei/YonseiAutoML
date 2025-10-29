import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

'''
From https://github.com/hw-ch0/IPMTL/blob/35009698edfcbe2893c04a1738505e60a62be7c5/im2im_pred/utils.py
 
if index==0:
    # w_semantic, w_depth, w_normal = 1/3, 1/3, 1/3
    weights[index,:] = 1/3, 1/3, 1/3
else:
    loss_prev = weights[index-1,0]*avg_cost[index-1,0] + weights[index-1,1]*avg_cost[index-1,3] + weights[index-1,2]*avg_cost[index-1,6]
    weights[index,:] = (loss_prev/avg_cost[index-1,0])/3, (loss_prev/avg_cost[index-1,3])/3, (loss_prev/avg_cost[index-1,6])/3
    if not index==1:
        loss_prev2 = weights[index-2,0]*avg_cost[index-2,0] + weights[index-2,1]*avg_cost[index-2,3] + weights[index-2,2]*avg_cost[index-2,6]
        difficulties[index,0] = (avg_cost[index-1,0]/avg_cost[index-2,0]) / (loss_prev/loss_prev2)
        difficulties[index,1] = (avg_cost[index-1,3]/avg_cost[index-2,3]) / (loss_prev/loss_prev2)
        difficulties[index,2] = (avg_cost[index-1,6]/avg_cost[index-2,6]) / (loss_prev/loss_prev2)
'''

class LSBwoD(AbsWeighting):
    r"""Loss Scale Balancing (LSB).
    
    """
    def __init__(self):
        super(LSBwoD, self).__init__()
        
    def backward(self, losses, **kwargs):
        if not hasattr(self, 'prev_weight'):
            self.prev_weight = torch.ones_like(losses).detach() / self.task_num
            self.loss_cache = 0
            self.losses_cache = 0
            self.iter = 0
        
        loss = torch.mul(losses, self.prev_weight).sum()
        self.loss_cache += loss.detach() / self.train_batch
        self.losses_cache += losses.detach() / self.train_batch
        self.iter += 1
        if (self.iter+1) % self.train_batch==0:
            self.prev_weight = self.loss_cache / (self.losses_cache * self.task_num)
            self.loss_cache = 0
            self.losses_cache = 0
            
        loss.backward()
        return self.prev_weight.detach().cpu().numpy()