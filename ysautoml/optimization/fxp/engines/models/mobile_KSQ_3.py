import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
import pdb
import numpy as np
from collections import OrderedDict

__all__ = ['MobileNetV2', 'mobilenet_v2_KSQ_3']

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class absol(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.abs()

    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = torch.sign(input)
        grad_input = grad_input + 1
        grad_input = ((grad_input+1e-6)/2).round()
        grad_input = (2*grad_input) - 1
#        grad_input[input==0] = 1
        return grad_output * grad_input

class RoundWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, g):
        return g

class DSQConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                momentum = 0.1,
                num_bit = 4, QInput = True, bSetQ = True, symmetric=True):
        super(DSQConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bit = num_bit
        self.quan_input = QInput
        self.symmetric = symmetric
        self.bit_range = 2**self.num_bit -1
        self.is_quan = bSetQ
        self.temp = -1
        self.q_value = torch.from_numpy(np.linspace(0,1,2))
        self.q_value = self.q_value.reshape(len(self.q_value),1,1,1,1).float()

        if self.is_quan:
            # using int32 max/min as init and backprogation to optimization
            # Weight
            self.uW = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
            self.lW  = nn.Parameter(data = torch.tensor((-1) * (2**32)).float())
            self.register_buffer('init', torch.tensor(0).float())
            self.beta = nn.Parameter(data = torch.tensor(0.2).float())

            # Activation input
            if self.quan_input:
                self.uA = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
                if self.symmetric:
                    self.lA = nn.Parameter(data = torch.tensor(0).float())

    def clipping(self, x, upper, lower):
        # clip lower
        x = x + F.relu(lower - x)
        # clip upper
        x = x - F.relu(x - upper)

        return x

    def step(self, x):
        if self.num_bit == 1:
            output = (x+1) / 2
            output = RoundWithGradient.apply(output)
            return 2 * output - 1
        else:
            return RoundWithGradient.apply(x)

    def quan(self, x, u, l):
        delta = (u - l) / (self.bit_range)
        interval = (x - l) / delta
        return 2*RoundWithGradient.apply(interval) - 1

    def w_soft_quan(self, x, u, l, sigma):
        bit_range = 15
        delta = (u - l) / bit_range
        interval = (x - l) / delta
        interval = torch.clamp(interval, min=0, max=bit_range)

        # Following the LSQ method
        output = 2 * self.w_soft_argmax(interval, 2, 1) - bit_range
        return output / bit_range

    def w_soft_argmax(self, x, T, sigma):
        x_floor = x.floor()
        x = x - x_floor.detach()
        # Distance score
        m_p = torch.exp(-absol.apply(x.unsqueeze(0).repeat(len(self.q_value.cuda()),1,1,1,1)
                                     - self.q_value.cuda()))

        # Get the kernel value
        max_value, max_idx = m_p.max(dim=0)
        max_idx = max_idx.unsqueeze(0).float().cuda()
        k_p = torch.exp(-(torch.pow(self.q_value.cuda()-max_idx, 2).float()/(sigma**2)))

        # Get the score
        score = m_p * k_p

        # Flexible temperature
        denorm = (score[0] - score[1]).abs()
        T_ori = T
        T = T / denorm
        T = T.detach()

        tmp_score = T * score

        prob = torch.exp(tmp_score)
        denorm = prob.sum(dim=0, keepdim=True)
        prob = prob / denorm

        q_var = self.q_value.clone()
        q_var[0] = q_var[0] - (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        q_var[1] = q_var[1] + (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        output = (q_var.cuda() * prob).sum(dim=0)
        output = output + x_floor

        return output

    def a_soft_quan(self, x, u, l, sigma):
        bit_range = 15
        delta = (u - l) / bit_range
        interval = (x - l) / delta
        interval = torch.clamp(interval, min=0, max=bit_range)
        output = self.a_soft_argmax(interval, 2, 2)
        if self.symmetric:
            output = 2 * output - bit_range
        return output / bit_range

    def a_soft_argmax(self, x, T, sigma):
        x_floor = x.floor()
        x = x - x_floor.detach()
        # Distance score
        m_p = torch.exp(-absol.apply(x.unsqueeze(0).repeat(len(self.q_value.cuda()),1,1,1,1)
                                     - self.q_value.cuda()))

        # Get the kernel value
        max_value, max_idx = m_p.max(dim=0)
        max_idx = max_idx.unsqueeze(0).float().cuda()
        k_p = torch.exp(-(torch.pow(self.q_value.cuda()-max_idx, 2).float()/(sigma**2)))

        # Get the score
        score = m_p * k_p

        # Flexible temperature
        denorm = (score[0] - score[1]).abs()
        T_ori = T
        T = T / denorm
        T = T.detach()

        tmp_score = T * score

        prob = torch.exp(tmp_score)
        denorm = prob.sum(dim=0, keepdim=True)
        prob = prob / denorm

        q_var = self.q_value.clone()
        q_var[0] = q_var[0] - (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        q_var[1] = q_var[1] + (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        output = (q_var.cuda() * prob).sum(dim=0)
        output = output + x_floor

        return output

    def forward(self, x):
        if self.is_quan:
            if self.init == 1:
                print(self.init)
                # self.init = torch.tensor(0)
                self.lW.data = torch.tensor(-3.0)
                self.uW.data = torch.tensor(3.0)
                if self.quan_input:
                    self.uA.data = x.std() * 3

            # For softdown the degree of discrete
            if self.training:
                self.temp = self.temp + 1

            sigma = 4
            curr_running_lw = self.lW
            curr_running_uw = self.uW

            if self.quan_input:
                curr_running_la = 0
                curr_running_ua = self.uA
                if self.symmetric==True:
                    curr_running_la = self.lA


            # Weight kernel_soft_argmax
            mean = self.weight.data.mean()
            std = self.weight.data.std()
            norm_weight = self.weight.add(-mean).div(std)
            Qweight = self.w_soft_quan(norm_weight, curr_running_uw, curr_running_lw, sigma)

            Qbias = self.bias

            # Input(Activation)
            Qactivation = x
            if self.quan_input:
                Qactivation = self.a_soft_quan(Qactivation, curr_running_ua, curr_running_la, sigma)

            if self.init == 1:
                # print(self.init)
                self.init = torch.tensor(0)
                q_output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
                ori_output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

                self.beta.data = torch.mean(torch.abs(ori_output)) / \
                                 torch.mean(torch.abs(q_output))

            output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
            output = torch.abs(self.beta) * output

        else:
            output =  F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

        return output

class DSQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                num_bit = 4, bSetQ = True, QInput=True):
        super(DSQLinear, self).__init__(in_features, out_features, bias)
        self.num_bit = num_bit
        self.quan_input = QInput
        self.bit_range = 2**self.num_bit -1
        self.is_quan = bSetQ
        self.temp = -1
        self.q_value = torch.from_numpy(np.linspace(0,1,2))
        self.q_value = self.q_value.reshape(len(self.q_value),1,1).float()

        if self.is_quan:
            # using int32 max/min as init and backprogation to optimization
            # Weight
            self.uW = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
            self.lW  = nn.Parameter(data = torch.tensor((-1) * (2**32)).float())
            self.register_buffer('init', torch.tensor(0).float())
            self.beta = nn.Parameter(data = torch.tensor(0.2).float())

            # Activation input
            if self.quan_input:
                self.uA = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
                # self.lA = nn.Parameter(data = torch.tensor(0).float())

    def clipping(self, x, upper, lower):
        # clip lower
        x = x + F.relu(lower - x)
        # clip upper
        x = x - F.relu(x - upper)

        return x

    def step(self, x):
        if self.num_bit == 1:
            output = (x+1) / 2
            output = RoundWithGradient.apply(output)
            return 2 * output - 1
        else:
            return RoundWithGradient.apply(x)

    def quan(self, x, u, l):
        delta = (u - l) / (self.bit_range)
        interval = (x - l) / delta
        return 2*RoundWithGradient.apply(interval) - 1

    def w_soft_quan(self, x, u, l, sigma):
        bit_range = 15
        delta = (u - l) / bit_range
        interval = (x - l) / delta
        interval = torch.clamp(interval, min=0, max=bit_range)

        # Following the LSQ method
        output = 2 * self.w_soft_argmax(interval, 2, 1) - bit_range
        return output / bit_range

    def w_soft_argmax(self, x, T, sigma):
        x_floor = x.floor()
        x = x - x_floor.detach()
        # Distance score
        m_p = torch.exp(-absol.apply(x.unsqueeze(0).repeat(len(self.q_value.cuda()),1,1)
                                     - self.q_value.cuda()))

        # Get the kernel value
        max_value, max_idx = m_p.max(dim=0)
        max_idx = max_idx.unsqueeze(0).float().cuda()
        k_p = torch.exp(-(torch.pow(self.q_value.cuda()-max_idx, 2).float()/(sigma**2)))

        # Get the score
        score = m_p * k_p

        # Flexible temperature
        denorm = (score[0] - score[1]).abs()
        T_ori = T
        T = T / denorm
        T = T.detach()

        tmp_score = T * score

        prob = torch.exp(tmp_score)
        denorm = prob.sum(dim=0, keepdim=True)
        prob = prob / denorm

        q_var = self.q_value.clone()
        q_var[0] = q_var[0] - (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        q_var[1] = q_var[1] + (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        output = (q_var.cuda() * prob).sum(dim=0)
        output = output + x_floor

        return output

    def a_soft_quan(self, x, u, l, sigma):
        bit_range = 15
        delta = (u - l) / bit_range
        interval = (x - l) / delta
        interval = torch.clamp(interval, min=0, max=bit_range)
        output = self.a_soft_argmax(interval, 2, 2)
        return output / bit_range

    def a_soft_argmax(self, x, T, sigma):
        x_floor = x.floor()
        x = x - x_floor.detach()
        # Distance score
        m_p = torch.exp(-absol.apply(x.unsqueeze(0).repeat(len(self.q_value.cuda()),1,1)
                                     - self.q_value.cuda()))

        # Get the kernel value
        max_value, max_idx = m_p.max(dim=0)
        max_idx = max_idx.unsqueeze(0).float().cuda()
        k_p = torch.exp(-(torch.pow(self.q_value.cuda()-max_idx, 2).float()/(sigma**2)))

        # Get the score
        score = m_p * k_p

        # Flexible temperature
        denorm = (score[0] - score[1]).abs()
        T_ori = T
        T = T / denorm
        T = T.detach()

        tmp_score = T * score

        prob = torch.exp(tmp_score)
        denorm = prob.sum(dim=0, keepdim=True)
        prob = prob / denorm

        q_var = self.q_value.clone()
        q_var[0] = q_var[0] - (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        q_var[1] = q_var[1] + (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        output = (q_var.cuda() * prob).sum(dim=0)
        output = output + x_floor

        return output

    def forward(self, x):

        if self.is_quan:
            if self.init == 1:
                print(self.init)
                # self.init = torch.tensor(0)
                self.lW.data = torch.tensor(-3.0)
                self.uW.data = torch.tensor(3.0)
                self.uA.data = x.std() * 3

            # For softdown the degree of discrete
            if self.training:
                self.temp = self.temp + 1

            sigma = 4
            curr_running_lw = self.lW
            curr_running_uw = self.uW

            if self.quan_input:
                curr_running_la = 0
                curr_running_ua = self.uA

            # Weight kernel_soft_argmax
            mean = self.weight.data.mean()
            std = self.weight.data.std()
            norm_weight = self.weight.add(-mean).div(std)
            Qweight = self.w_soft_quan(norm_weight, curr_running_uw, curr_running_lw, sigma)

            Qbias = self.bias

            # Input(Activation)
            Qactivation = x
            if self.quan_input:
                Qactivation = self.a_soft_quan(Qactivation, curr_running_ua, curr_running_la, sigma)

            if self.init == 1:
                self.init = torch.tensor(0)
                q_output = F.linear(Qactivation, Qweight, Qbias)
                ori_output = F.linear(x, self.weight, self.bias)

                self.beta.data = torch.mean(torch.abs(ori_output)) / \
                                 torch.mean(torch.abs(q_output))

            output = F.linear(Qactivation, Qweight, Qbias)
            output = torch.abs(self.beta) * output

        else:
            output =  F.linear(x, self.weight, self.bias)

        return output

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None, initial=False, symmetric=False):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if initial:
            QInput=False
        else:
            QInput=True

        super(ConvBNReLU, self).__init__(
            DSQConv(in_planes, out_planes, kernel_size, stride, padding, groups=groups,
                      bias=False, QInput=QInput, symmetric=symmetric),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None, first_layer=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        symmetric=True

        if first_layer:
            symmetric=False

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, symmetric=symmetric)) # a

        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer, symmetric=False),
            # pw-linear
            DSQConv(hidden_dim, oup, 1, 1, 0, bias=False, symmetric=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 num_bit=4,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer, initial=True)] ##### initial
        # building inverted residual blocks
        first_layer = True
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, first_layer=first_layer))
                input_channel = output_channel
                first_layer=False
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, symmetric=True))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            DSQLinear(self.last_channel, num_classes),
        )

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        ret_dict = dict()
        ret_dict['out'] = self._forward_impl(x)
        return ret_dict


def mobilenet_v2_KSQ_3(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
#    if pretrained:
#        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
#                                              progress=progress)
#        model.load_state_dict(state_dict)
    return model
