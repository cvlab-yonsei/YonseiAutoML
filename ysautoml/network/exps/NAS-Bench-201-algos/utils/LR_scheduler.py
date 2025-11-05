import torch
import torch.nn as nn



class AdaptiveParamSchedule(torch.optim.lr_scheduler.LambdaLR):

    def __init__(self, optimizer,epochs, last_epoch=-1, eta_min = 0):
        self.exp_coeff = 1
        self.cur_ep = 0

        def lr_lambda(step):
            y = ((-self.cur_ep+epochs)) ** self.exp_coeff / (epochs ** self.exp_coeff/ (1 - eta_min)) + eta_min
            return float(y)
        super(AdaptiveParamSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)
