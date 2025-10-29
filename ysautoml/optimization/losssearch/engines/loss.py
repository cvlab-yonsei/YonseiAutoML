import torch
import torch.nn as nn
import torch.nn.functional as F


class custom_loss(nn.Module):
    def __init__(self):
        super(custom_loss, self).__init__()
        self.theta = nn.Parameter(data = torch.tensor([ 1.7972,  1.2659,  1.5359,  0.7382,  1.2012,  0.6994, -0.2802,  0.0219,
        -0.1034,  0.1260, -0.2372, -0.2678]
))
#        

    def t_fun(self, x, theta, x_range = [-1, 1]):
        M = theta.size(0)// 2

        interval = (x_range[1] - x_range[0]) / M
        y = 0
        for m in range(M):
            if m == M - 1:
                ind = (x >= (x_range[0] + interval * m)).float()
            elif m == 0:
                ind = (x < (x_range[0] + interval * (m + 1))).float()
            else:
                ind = ((x >= (x_range[0] + interval * m)).float() * (x < (x_range[0] + interval * (m + 1))).float())
            y += (x * theta[m] + theta[m+M]) * ind

        return y

    def forward(self, cos_x, y):
        M = self.theta.size(0)// 2
        theta = self.theta
        y_onehot = F.one_hot(y, 10)

        t1_cos_x = self.t_fun(cos_x, theta[:2 * M], x_range=torch.Tensor([-1, 1]).to(cos_x.device))
        # t2_cos_x = t_fun(cos_x, theta[2*M:], x_range=torch.Tensor([-1,1]).to(cos_x.device))
        # p_y_t = F.softmax(t1_cos_x * y_onehot + t2_cos_x * (1 - y_onehot), dim=1)[torch.arange(cos_x.size(0)).to(cos_x.device), y]

        # tao_p_y_t = t_fun(p_y_t, theta[2 * M:], x_range=torch.Tensor([0, 1]).to(cos_x.device))
        loss = -F.log_softmax((t1_cos_x * y_onehot + cos_x * (1 - y_onehot)), dim=1)[
            torch.arange(cos_x.size(0)).to(cos_x.device), y]
        return loss.mean()