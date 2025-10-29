import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class AMTL(AbsWeighting):
    r"""Achievement-based Multi-task Learning (AMTL).
    
    This method is proposed in `Achievement-based Training Progress Balancing for Multi-Task Learning (ICCV 2023) <https://openaccess.thecvf.com/content/ICCV2023/html/Yun_Achievement-Based_Training_Progress_Balancing_for_Multi-Task_Learning_ICCV_2023_paper.html>`_ \
    and implemented by us.

    """
    def __init__(self):
        super(AMTL, self).__init__()
        
    def backward(self, losses, **kwargs):
        # Load hyperparameters
        if not hasattr(self, 'potentials'): 
            self.potentials = eval(kwargs['potentials_' + kwargs['dataset_str']]) 
            assert isinstance(self.potentials, list), 'TypeError: type of potentials should be List.'
            self.potentials = torch.Tensor(self.potentials).to(self.device) * 1.05 # multiply a slight margin
            self.focusing_factor = kwargs['focusing_factor'] 

        # Geometric at first epoch
        if not hasattr(self, 'val_results'): 
            weight = torch.ones_like(losses).to(self.device)
            
        else:
            # Given validation results (self.model.val_results), 
            def get_achievement(cur_results):
                cur_achievement = [1] * self.task_num
                for tn, task in enumerate(self.task_name):
                    for i, weight in enumerate(self.task_dict[task]['weight']): # i: metric number
                        cur_achievement[tn] *= cur_results[task][i] ** (2*weight-1) # (1,0) -> (1,-1)
                    cur_achievement[tn] = cur_achievement[tn] ** (1/len(self.task_dict[task]['weight']))
                return torch.Tensor(cur_achievement).unsqueeze(1).to(self.device)
            
            cur_achievement = get_achievement(self.val_results)
            weight = torch.pow(1 - cur_achievement / self.potentials, self.focusing_factor)
            weight = torch.softmax(weight, dim=0)
        
        # loss = torch.pow( torch.pow(losses, weight).prod(), 1./self.task_num) # GM
        loss = torch.mul(losses, weight).sum() # AM
        # p = 2
        # loss = torch.pow( (torch.mul(torch.pow(losses, p), weight)).sum() / self.task_num, 1/p) # QM
        loss.backward()
        batch_weight = losses / (self.task_num * losses.prod())
        return batch_weight.detach().cpu().numpy()