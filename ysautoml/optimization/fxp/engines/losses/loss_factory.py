from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class RoundWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, g):
        return g

class CeilWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.ceil()
    @staticmethod
    def backward(ctx, g):
        return g

class FloorWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.floor()
    @staticmethod
    def backward(ctx, g):
        return g

def cross_entropy_dist_epoch(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)

    def loss_fn(outputs, outputs_f, labels, epoch, **_):
        loss_dict = dict()
        full_gt_loss = cross_entropy_fn(outputs_f['out'], labels)
        gt_loss = cross_entropy_fn(outputs['out'], labels)
        dist_loss = 0
        layer_names = outputs.keys()
        len_layer = len(layer_names)

        for i, layer_name in enumerate(layer_names):
            if i == len_layer - 1:
                continue
            dist_loss += l1_fn(outputs[layer_name], outputs_f[layer_name])

        scale = epoch / 100
        if epoch == 100:
            scale = 1

        loss_dict['loss'] = scale*(gt_loss + dist_loss) + full_gt_loss
        loss_dict['gt_loss'] = gt_loss
        loss_dict['full_gt_loss'] = full_gt_loss
        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def cross_entropy_dist(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)

    def loss_fn(outputs, outputs_f, labels, **_):
        loss_dict = dict()
        full_gt_loss = cross_entropy_fn(outputs_f['out'], labels)
        gt_loss = cross_entropy_fn(outputs['out'], labels)
        dist_loss = 0
        layer_names = outputs.keys()
        len_layer = len(layer_names)

        for i, layer_name in enumerate(layer_names):
            if i == len_layer - 1:
                continue
            dist_loss += l1_fn(outputs[layer_name], outputs_f[layer_name])

        loss_dict['loss'] = gt_loss + dist_loss + full_gt_loss
        loss_dict['gt_loss'] = gt_loss
        loss_dict['full_gt_loss'] = full_gt_loss
        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def cross_entropy(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)

    def loss_fn(outputs, labels, **_):
        loss_dict = dict()
        print(outputs['out'])
        gt_loss = cross_entropy_fn(outputs['out'], labels)
        # gt_loss = cross_entropy_fn(outputs, labels)
        loss_dict['loss'] = gt_loss
        loss_dict['gt_loss'] = gt_loss
        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def cross_entropy_t_reg(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)

    def loss_fn(outputs, labels, **_):
        loss_dict = dict()
        pred = outputs['out']
        gt_loss = cross_entropy_fn(pred, labels)

        target_bit = torch.tensor(2).cuda()
        bit_level = outputs['bit_level']
        param_num = outputs['param_num']

        layers = np.shape(bit_level)[0]
        total_num = 0
        total_bit_num = 0
        for i in range(layers):
            total_num += param_num[i]
            total_bit_num += (param_num[i] * bit_level[i])

        pred_bit = total_bit_num / total_num
        t_reg_loss = l1_fn(pred_bit, target_bit)

        loss_dict['loss'] = gt_loss + t_reg_loss
        # loss_dict['loss'] = gt_loss
#        loss_dict['loss'] = gt_loss
        loss_dict['t_reg_loss'] = t_reg_loss
        loss_dict['gt_loss'] = gt_loss
        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def cross_entropy_t_reg_l1_reg(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)

    def loss_fn(outputs, labels, epoch, **_):
        loss_dict = dict()
        pred = outputs['out']
        gt_loss = cross_entropy_fn(pred, labels)

        target_bit = torch.tensor(2).cuda()
        reg_bit = torch.tensor(1).cuda()
        bit_level = outputs['bit_level']
        bit_level_int = bit_level.copy()
        param_num = outputs['param_num']

        layers = np.shape(bit_level)[0]
        total_num = 0
        total_bit_num = 0
        total_num_b = 0
        total_bit_num_b = 0
        bit_loss = 0
        for i in range(layers):
            # bit_level_int[i] = RoundWithGradient.apply(bit_level[i])
            # bit_loss += (bit_level[i]).abs()
            total_num += param_num[i]
            total_bit_num += (param_num[i] * bit_level[i])
            total_num_b += 1
            total_bit_num_b += bit_level[i]

        pred_bit = total_bit_num / total_num
        pred_bit_b = total_bit_num_b / total_num_b
        t_reg_loss = l1_fn(pred_bit, target_bit)
        # bit_loss = (0)* bit_loss / layers
        bit_loss = (1e-1) * l1_fn(pred_bit_b, reg_bit)

        loss_dict['loss'] = gt_loss + t_reg_loss + bit_loss
        # loss_dict['loss'] = gt_loss
#        loss_dict['loss'] = gt_loss
        loss_dict['t_reg_loss'] = t_reg_loss
        loss_dict['bit_loss'] = bit_loss
        loss_dict['gt_loss'] = gt_loss
        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def cross_entropy_t_reg_l1_cr_reg(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)

    def loss_fn(outputs, labels, **_):
        loss_dict = dict()
        pred = outputs['out']
        gt_loss = cross_entropy_fn(pred, labels)

        target_bit = torch.tensor(2).cuda()
        bit_level = outputs['bit_level']
        param_num = outputs['param_num']
        weight_cr = outputs['weight_cr']

        layers = np.shape(bit_level)[0]
        total_num = 0
        total_bit_num = 0
        bit_loss = 0
        for i in range(layers):
            bit_loss += (bit_level[i] - 2).abs() * torch.exp(-weight_cr[i])
            total_num += param_num[i]
            total_bit_num += (param_num[i] * bit_level[i])

        pred_bit = total_bit_num / total_num
        t_reg_loss = (0.8) * l1_fn(pred_bit, target_bit)
        bit_loss = (0.2) * bit_loss / layers

        loss_dict['loss'] = gt_loss + t_reg_loss + bit_loss
        # loss_dict['loss'] = gt_loss
#        loss_dict['loss'] = gt_loss
        loss_dict['t_reg_loss'] = t_reg_loss
        loss_dict['bit_loss'] = bit_loss
        loss_dict['gt_loss'] = gt_loss
        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def cross_entropy_t_reg_l1_cr_reg_1(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)

    def loss_fn(outputs, labels, **_):
        loss_dict = dict()
        cr_dict = dict()
        pred = outputs['out']
        gt_loss = cross_entropy_fn(pred, labels)

        target_bit = torch.tensor(2).cuda()
        bit_level = outputs['bit_level']
        param_num = outputs['param_num']
        weight_cr = outputs['weight_cr']

        layers = np.shape(bit_level)[0]
        total_num = 0
        total_bit_num = 0
        bit_loss = 0

        for i in range(layers):
            cr_dict['{}'.format(bit_level[i].round())] = weight_cr[i]

        for i in range(layers):
            if cr_dict['{}'.format(bit_level[i].round())] < weight_cr[i]:
                cr_dict['{}'.format(bit_level[i].round())] = weight_cr[i]
            total_num += param_num[i]
            total_bit_num += (param_num[i] * bit_level[i])

        for i in range(layers):
            weight_cr[i] = weight_cr[i] / cr_dict['{}'.format(bit_level[i].round())]
            # bit_loss += (bit_level[i] - 1).abs() * torch.exp(-(weight_cr[i]))
            bit_loss += (bit_level[i] - 1).abs() * torch.exp(-(weight_cr[i]))

        pred_bit = total_bit_num / total_num
        t_reg_loss = l1_fn(pred_bit, target_bit)
        bit_loss = (1e-1) * bit_loss / layers

        loss_dict['loss'] = gt_loss + t_reg_loss + bit_loss
        # loss_dict['loss'] = gt_loss
#        loss_dict['loss'] = gt_loss
        loss_dict['t_reg_loss'] = t_reg_loss
        loss_dict['bit_loss'] = bit_loss
        loss_dict['gt_loss'] = gt_loss
        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def cross_entropy_t_reg_l1_cr_reg_2(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)

    def loss_fn(outputs, labels, epoch, **_):
        loss_dict = dict()
        cr_dict = dict()
        pred = outputs['out']
        gt_loss = cross_entropy_fn(pred, labels)

        target_bit = torch.tensor(2).cuda()
        reg_bit = torch.tensor(1).cuda()
        bit_level = outputs['bit_level']
        param_num = outputs['param_num']
        weight_cr = outputs['weight_cr']

        layers = np.shape(bit_level)[0]
        total_num = 0
        total_bit_num = 0
        total_num_cr = 0
        total_bit_num_cr = 0
        bit_loss = 0

        for i in range(layers):
            cr_dict['{}'.format(bit_level[i].round())] = weight_cr[i]

        for i in range(layers):
            if cr_dict['{}'.format(bit_level[i].round())] < weight_cr[i]:
                cr_dict['{}'.format(bit_level[i].round())] = weight_cr[i]
            total_num += param_num[i]
            total_bit_num += (param_num[i] * bit_level[i])

        # cr_power = (epoch / 320.) * 2
        cr_power = 2

        for i in range(layers):
            weight_cr[i] = weight_cr[i] / cr_dict['{}'.format(bit_level[i].round())]
            # bit_loss += (bit_level[i] - 1).abs() * torch.exp(-(weight_cr[i]))
            # bit_loss += (bit_level[i]).abs() * torch.exp(-(weight_cr[i] ** cr_power))
#            total_num_cr += param_num[i] * (1-(weight_cr[i] ** cr_power))
#            total_bit_num_cr += (param_num[i] * bit_level[i]) * (1-(weight_cr[i] ** cr_power))
#            total_num_cr += (1-(weight_cr[i] ** cr_power))
#            total_bit_num_cr += bit_level[i] * (1-(weight_cr[i] ** cr_power))
            total_num_cr += torch.exp(-weight_cr[i])
            total_bit_num_cr += bit_level[i] * torch.exp(-weight_cr[i])

#            bit_loss += (bit_level[i]).abs() * (1-(weight_cr[i] ** cr_power))

        pred_bit = total_bit_num / total_num
        pred_bit_cr = total_bit_num_cr / (total_num_cr + 1e-6)
        t_reg_loss = l1_fn(pred_bit, target_bit)
        bit_loss = (1e-1) * l1_fn(pred_bit_cr, reg_bit)
        # bit_loss = (1e-1) * bit_loss / layers

        loss_dict['loss'] = gt_loss + t_reg_loss + bit_loss
        # loss_dict['loss'] = gt_loss
#        loss_dict['loss'] = gt_loss
        loss_dict['t_reg_loss'] = t_reg_loss
        loss_dict['bit_loss'] = bit_loss
        loss_dict['gt_loss'] = gt_loss
        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def cross_entropy_t_reg_l1_cr_reg_3(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)

    def loss_fn(outputs, labels, epoch, **_):
        loss_dict = dict()
        cr_dict = dict()
        pred = outputs['out']
        gt_loss = cross_entropy_fn(pred, labels)

        target_bit = torch.tensor(2).cuda()
        reg_bit = torch.tensor(1).cuda()
        bit_level = outputs['bit_level']
        param_num = outputs['param_num']
        weight_cr = outputs['weight_cr']

        layers = np.shape(bit_level)[0]
        total_num = 0
        total_bit_num = 0
        total_num_cr = 0
        total_bit_num_cr = 0
        bit_loss = 0

        for i in range(layers):
            cr_dict['{}'.format(bit_level[i].round())] = weight_cr[i]

        for i in range(layers):
            if cr_dict['{}'.format(bit_level[i].round())] < weight_cr[i]:
                cr_dict['{}'.format(bit_level[i].round())] = weight_cr[i]
            total_num += param_num[i]
            total_bit_num += (param_num[i] * bit_level[i])

        # cr_power = (epoch / 320.) * 2
        cr_weight = (epoch / 320) * 2
        if cr_weight >=1:
            cr_weight = 1
        else:
            cr_weight = 0
#        cr_weight = 1

        cr_power = 2

        for i in range(layers):
            weight_cr[i] = weight_cr[i] / cr_dict['{}'.format(bit_level[i].round())]
            # bit_loss += (bit_level[i] - 1).abs() * torch.exp(-(weight_cr[i]))
            # bit_loss += (bit_level[i]).abs() * torch.exp(-(weight_cr[i] ** cr_power))
            total_num_cr += param_num[i] * (1-(weight_cr[i] ** cr_power))
            total_bit_num_cr += (param_num[i] * bit_level[i]) * (1-(weight_cr[i] ** cr_power))
#            total_num_cr += (1-(weight_cr[i] ** cr_power))
#            total_bit_num_cr += bit_level[i] * (1-(weight_cr[i] ** cr_power))
#            total_num_cr += torch.exp(-(weight_cr[i] ** cr_power))
#            total_bit_num_cr += bit_level[i] * torch.exp(-(weight_cr[i] ** cr_power))

#            bit_loss += (bit_level[i]).abs() * (1-(weight_cr[i] ** cr_power))

        pred_bit = total_bit_num / total_num
        pred_bit_cr = total_bit_num_cr / total_num_cr
        t_reg_loss = l1_fn(pred_bit, target_bit)
        bit_loss = (1e-1) * cr_weight * l1_fn(pred_bit_cr, reg_bit)
        # bit_loss = (1e-1) * bit_loss / layers

        loss_dict['loss'] = gt_loss + t_reg_loss + bit_loss
        # loss_dict['loss'] = gt_loss
#        loss_dict['loss'] = gt_loss
        loss_dict['t_reg_loss'] = t_reg_loss
        loss_dict['bit_loss'] = bit_loss
        loss_dict['gt_loss'] = gt_loss
        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def cross_entropy_t_reg_l1_cr_reg_4(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)
    l2_fn = torch.nn.MSELoss(reduction=reduction)

    def loss_fn(outputs, labels, epoch, **_):
        loss_dict = dict()
        cr_dict = dict()
        idx_dict = dict()
        pred = outputs['out']
        gt_loss = cross_entropy_fn(pred, labels)

        target_bit = torch.tensor(2).cuda()
        bit_level = outputs['bit_level']
        param_num = outputs['param_num']
        weight_cr = outputs['weight_cr']

        layers = np.shape(bit_level)[0]
        total_num = 0
        total_bit_num = 0

        for i in range(layers):
            cr_dict['{}'.format(bit_level[i])] = 0
            idx_dict['{}'.format(bit_level[i])] = 0

        for i in range(layers):
            cr_dict['{}'.format(bit_level[i])] += torch.exp(weight_cr[i])
            idx_dict['{}'.format(bit_level[i])] += 1
            total_num += param_num[i]
            total_bit_num += (param_num[i] * bit_level[i])

        for i in cr_dict.keys():
            cr_dict[i] = cr_dict[i] / idx_dict[i]

        pred_bit = total_bit_num / total_num
        t_reg_loss = l1_fn(pred_bit, target_bit)
        # bit_loss = (1e-1) * bit_loss / layers

        loss_dict['loss'] = gt_loss + t_reg_loss
        loss_dict['t_reg_loss'] = t_reg_loss
        loss_dict['pred_bit'] = pred_bit
        loss_dict['gt_loss'] = gt_loss
        return loss_dict, cr_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def cross_entropy_t_reg_l1_cr_reg_5(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)
    l2_fn = torch.nn.MSELoss(reduction=reduction)

    def loss_fn(outputs, labels, epoch, **_):
        loss_dict = dict()
        cr_dict = dict()
        idx_dict = dict()
        pred = outputs['out']
        gt_loss = cross_entropy_fn(pred, labels)

        target_bit = torch.tensor(2).cuda()
        bit_level = outputs['bit_level']
        param_num = outputs['param_num']
        weight_cr = outputs['weight_cr']

        layers = np.shape(bit_level)[0]
        total_num = 0
        total_bit_num = 0

        for i in range(layers):
            cr_dict['{}'.format(bit_level[i])] = 0
            idx_dict['{}'.format(bit_level[i])] = 0

        for i in range(layers):
            cr_dict['{}'.format(bit_level[i])] += torch.exp(weight_cr[i])
            idx_dict['{}'.format(bit_level[i])] += 1
            total_num += param_num[i]
            total_bit_num += (param_num[i] * bit_level[i])

        for i in cr_dict.keys():
            cr_dict[i] = cr_dict[i] / idx_dict[i]

        pred_bit = total_bit_num / total_num
        t_reg_loss = l1_fn(pred_bit, target_bit)
        # bit_loss = (1e-1) * bit_loss / layers

        loss_dict['loss'] = gt_loss
        loss_dict['pred_bit'] = pred_bit
        loss_dict['gt_loss'] = gt_loss
        return loss_dict, cr_dict

    def bit_fn(outputs, labels, epoch, **_):
        loss_dict = dict()
def cross_entropy_t_reg_l1_cr_reg_5(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)
    l2_fn = torch.nn.MSELoss(reduction=reduction)

    def loss_fn(outputs, labels, epoch, **_):
        loss_dict = dict()
        cr_dict = dict()
        idx_dict = dict()
        pred = outputs['out']
        gt_loss = cross_entropy_fn(pred, labels)

        target_bit = torch.tensor(2).cuda()
        bit_level = outputs['bit_level']
        param_num = outputs['param_num']
        weight_cr = outputs['weight_cr']

        layers = np.shape(bit_level)[0]
        total_num = 0
        total_bit_num = 0

        for i in range(layers):
            cr_dict['{}'.format(bit_level[i])] = 0
            idx_dict['{}'.format(bit_level[i])] = 0

        for i in range(layers):
            cr_dict['{}'.format(bit_level[i])] += torch.exp(weight_cr[i])
            idx_dict['{}'.format(bit_level[i])] += 1
            total_num += param_num[i]
            total_bit_num += (param_num[i] * bit_level[i])

        for i in cr_dict.keys():
            cr_dict[i] = cr_dict[i] / idx_dict[i]

        pred_bit = total_bit_num / total_num
        t_reg_loss = l1_fn(pred_bit, target_bit)
        # bit_loss = (1e-1) * bit_loss / layers

        loss_dict['loss'] = gt_loss
        loss_dict['pred_bit'] = pred_bit
        loss_dict['gt_loss'] = gt_loss
        return loss_dict, cr_dict

    def bit_fn(outputs, labels, epoch, **_):
        loss_dict = dict()
        cr_dict = dict()
        idx_dict = dict()
        pred = outputs['out']
        gt_loss = cross_entropy_fn(pred, labels)

        target_bit = torch.tensor(2).cuda()
        bit_level = outputs['bit_level']
        param_num = outputs['param_num']
        weight_cr = outputs['weight_cr']

        layers = np.shape(bit_level)[0]
        total_num = 0
        total_bit_num = 0

        for i in range(layers):
            cr_dict['{}'.format(bit_level[i])] = 0
            idx_dict['{}'.format(bit_level[i])] = 0

        for i in range(layers):
            cr_dict['{}'.format(bit_level[i])] += torch.exp(weight_cr[i])
            idx_dict['{}'.format(bit_level[i])] += 1
            total_num += param_num[i]
            total_bit_num += (param_num[i] * bit_level[i])

        for i in cr_dict.keys():
            cr_dict[i] = cr_dict[i] / idx_dict[i]

        pred_bit = total_bit_num / total_num
        t_reg_loss = l1_fn(pred_bit, target_bit)
        # bit_loss = (1e-1) * bit_loss / layers

        loss_dict['t_reg_loss'] = t_reg_loss
        loss_dict['pred_bit'] = pred_bit
        return loss_dict, cr_dict

    return {'train':loss_fn,
            'bit_train':bit_fn,
            'val':cross_entropy_fn}

def cross_entropy_t_reg_l1_cr_reg_6(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)
    l2_fn = torch.nn.MSELoss(reduction=reduction)

    def loss_fn(outputs, labels, epoch, **_):
        loss_dict = dict()
        cr_dict = dict()
        idx_dict = dict()
        pred = outputs['out']
        gt_loss = cross_entropy_fn(pred, labels)

        target_bit = torch.tensor(2).cuda()
        bit_level = outputs['bit_level']
        param_num = outputs['param_num']
        weight_cr = outputs['weight_cr']

        layers = np.shape(bit_level)[0]
        total_num = 0
        total_bit_num = 0

        for i in range(layers):
            cr_dict['{}_{}'.format(bit_level[i], param_num[i])] = 0
            idx_dict['{}_{}'.format(bit_level[i], param_num[i])] = 0

        for i in range(layers):
            cr_dict['{}_{}'.format(bit_level[i], param_num[i])] += torch.exp(weight_cr[i])
            idx_dict['{}_{}'.format(bit_level[i], param_num[i])] += 1
            total_num += param_num[i]
            total_bit_num += (param_num[i] * bit_level[i])

        for i in cr_dict.keys():
            cr_dict[i] = cr_dict[i] / idx_dict[i]

        pred_bit = total_bit_num / total_num
        t_reg_loss = l1_fn(pred_bit, target_bit)
        # bit_loss = (1e-1) * bit_loss / layers

        loss_dict['loss'] = gt_loss
        loss_dict['pred_bit'] = pred_bit
        loss_dict['gt_loss'] = gt_loss
        return loss_dict, cr_dict

    def bit_fn(outputs, labels, epoch, **_):
        loss_dict = dict()
        cr_dict = dict()
        idx_dict = dict()
        pred = outputs['out']
        gt_loss = cross_entropy_fn(pred, labels)

        target_bit = torch.tensor(2).cuda()
        bit_level = outputs['bit_level']
        param_num = outputs['param_num']
        weight_cr = outputs['weight_cr']

        layers = np.shape(bit_level)[0]
        total_num = 0
        total_bit_num = 0

        for i in range(layers):
            cr_dict['{}'.format(bit_level[i])] = 0
            idx_dict['{}'.format(bit_level[i])] = 0

        for i in range(layers):
            cr_dict['{}'.format(bit_level[i])] += torch.exp(weight_cr[i])
            idx_dict['{}'.format(bit_level[i])] += 1
            total_num += param_num[i]
            total_bit_num += (param_num[i] * bit_level[i])

        for i in cr_dict.keys():
            cr_dict[i] = cr_dict[i] / idx_dict[i]

        pred_bit = total_bit_num / total_num
        t_reg_loss = l1_fn(pred_bit, target_bit)
        # bit_loss = (1e-1) * bit_loss / layers

        loss_dict['t_reg_loss'] = t_reg_loss
        loss_dict['pred_bit'] = pred_bit
        return loss_dict, cr_dict

    return {'train':loss_fn,
            'bit_train':bit_fn,
            'val':cross_entropy_fn}

def cross_entropy(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)

    def loss_fn(outputs, labels, **_):
        loss_dict = dict()
        gt_loss = cross_entropy_fn(outputs['out'], labels)
        # gt_loss = cross_entropy_fn(outputs, labels)
        loss_dict['loss'] = gt_loss
        loss_dict['gt_loss'] = gt_loss
        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def cross_entropy_t_reg_l1_cr_reg_7(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)
    l2_fn = torch.nn.MSELoss(reduction=reduction)

    def loss_fn(outputs, labels, **_):
        loss_dict = dict()
        cr_dict = dict()
        pred = outputs['out']
        gt_loss = cross_entropy_fn(pred, labels)

        target_bit = torch.tensor(2.0).cuda(outputs['gpu_n'])
        reg_bit = torch.tensor(1).cuda(outputs['gpu_n'])
        bit_level = outputs['bit_level'] # integer bit-width
        param_num = outputs['param_num']

        layers = np.shape(bit_level)[0]
        total_num = 0
        total_bit_num = 0
        real_bit_num = 0
        total_num_cr = 0
        total_bit_num_cr = 0
        bit_loss = 0

        for i in range(layers):
            total_num += param_num[i]
            total_bit_num += (param_num[i] * bit_level[i])
            real_bit_num += (param_num[i] * (bit_level[i].round()))

        pred_bit = total_bit_num / total_num
        pred_bit_real = real_bit_num / total_num

        # if pred_bit <= target_bit:
        #     t_reg_loss = torch.tensor(0.0)
        # else:
        #     t_reg_loss = l1_fn(pred_bit, target_bit)

        t_reg_loss = 0.1*l1_fn(pred_bit, target_bit)
        # t_reg_loss = torch.tensor(0.0)

        loss_dict['loss'] = gt_loss + t_reg_loss
        loss_dict['t_reg_loss'] = t_reg_loss
        loss_dict['gt_loss'] = gt_loss
        loss_dict['pred_bit'] = pred_bit_real
        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def cross_entropy_t_reg_l1_cr_reg_7_reg(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)
    l2_fn = torch.nn.MSELoss(reduction=reduction)

    def loss_fn(outputs, labels, **_):
        loss_dict = dict()
        cr_dict = dict()
        pred = outputs['out']
        gt_loss = cross_entropy_fn(pred, labels)

        target_bit = torch.tensor(2).cuda(outputs['gpu_n'])
        reg_bit = torch.tensor(1).cuda(outputs['gpu_n'])
        bit_level = outputs['bit_level'] # integer bit-width
        param_num = outputs['param_num']
        epoch = outputs['epoch']
        real_diff = outputs['real_diff']
        real_bit = outputs['real_bit']

        layers = np.shape(bit_level)[0]
        layers_diff = np.shape(real_diff)[0]

        total_num = 0
        total_bit_num = 0
        real_bit_num = 0

        total_layer = 0
        total_layer_diff = 0
        bit_loss = 0
        diff_loss = 0

        for i in range(layers):
            total_num += param_num[i]
            # total_bit_num += (param_num[i] * RoundWithGradient.apply(bit_level[i]))
            total_bit_num += (param_num[i] * bit_level[i])
            real_bit_num += (param_num[i] * (bit_level[i].round()))

            total_layer += 1
            bit_loss = bit_loss + torch.abs(bit_level[i] - bit_level[i].round())

        for i in range(layers_diff):
            total_layer_diff += 1
            weighted = (2 * torch.abs(real_bit[i].detach() - real_bit[i].round()))
            diff_loss = diff_loss + (weighted * real_diff[i])

        pred_bit = total_bit_num / total_num
        pred_bit_real = real_bit_num / total_num

        bit_loss = 0.0 * (bit_loss / total_layer) * (epoch / 135150)
        # diff_loss = 10 * (diff_loss / total_layer_diff) * (1 - (epoch / 135150))
        diff_loss = 10 * (diff_loss / total_layer_diff)

        # if pred_bit <= target_bit:
        #     t_reg_loss = torch.tensor(0.0)
        # else:
        #     t_reg_loss = l1_fn(pred_bit, target_bit)

        t_reg_loss = 0.5 * l1_fn(pred_bit, target_bit) + bit_loss
        # t_reg_loss = torch.tensor(0.0)

        loss_dict['loss'] = gt_loss + t_reg_loss
        loss_dict['t_reg_loss'] = t_reg_loss
        loss_dict['gt_loss'] = gt_loss
        loss_dict['pred_bit'] = pred_bit_real
        loss_dict['bit_loss'] = bit_loss
        loss_dict['diff_loss'] = diff_loss
        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def cross_entropy_t_reg_l1_cr_reg_8(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)
    l2_fn = torch.nn.MSELoss(reduction=reduction)

    def loss_fn(outputs, labels, **_):
        loss_dict = dict()
        cr_dict = dict()
        pred = outputs['out']
        gt_loss = cross_entropy_fn(pred, labels)

        target_bit = torch.tensor(2).cuda()
        target_bita = torch.tensor(3.5).cuda()
        reg_bit = torch.tensor(1).cuda()
        bit_level = outputs['bit_level'] # integer bit-width
        param_num = outputs['param_num']
        bita_level = outputs['bita_level']
        feat_num = outputs['feat_num']

        layers = np.shape(bit_level)[0]
        total_num = 0
        total_num_a = 0

        total_bit_num = 0
        total_bita_num = 0

        real_bit_num = 0
        real_bita_num = 0

        total_num_cr = 0
        total_bit_num_cr = 0
        bit_loss = 0

        for i in range(layers):
            total_num += param_num[i]
            total_bit_num += (param_num[i] * bit_level[i])
            real_bit_num += (param_num[i] * (bit_level[i].round()))

            total_num_a += param_num[i] * feat_num[i]
            total_bita_num += (param_num[i] * feat_num[i] * bit_level[i] * bita_level[i])
            real_bita_num += (param_num[i] * feat_num[i] * bit_level[i].round() * bita_level[i].round())


        pred_bit = total_bit_num / total_num
        pred_bit_real = real_bit_num / total_num

        pred_bita = total_bita_num / total_num_a
        pred_bita_real = real_bita_num / total_num_a

        # if pred_bit <= target_bit:
        #     t_reg_loss = torch.tensor(0.0)
        # else:
        #     t_reg_loss = l1_fn(pred_bit, target_bit)

        t_reg_loss = (0.1*l1_fn(pred_bit, target_bit)) + (0.02*l1_fn(pred_bita, target_bita))
        # t_reg_loss = 0.02*l1_fn(pred_bita, target_bita)

        loss_dict['loss'] = gt_loss + t_reg_loss
        loss_dict['t_reg_loss'] = t_reg_loss
        loss_dict['gt_loss'] = gt_loss
        loss_dict['pred_bit'] = pred_bit_real
        loss_dict['pred_bita'] = pred_bita_real

        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def cross_entropy_t_reg_l1_cr_reg_9(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)
    l2_fn = torch.nn.MSELoss(reduction=reduction)

    def loss_fn(outputs, labels, **_):
        loss_dict = dict()
        cr_dict = dict()
        pred = outputs['out']
        gt_loss = cross_entropy_fn(pred, labels)

        target_bit = torch.tensor(2).cuda()
        target_bita = torch.tensor(3.8).cuda()
        reg_bit = torch.tensor(1).cuda()
        bit_level = outputs['bit_level'] # integer bit-width
        param_num = outputs['param_num']
        bita_level = outputs['bita_level']
        feat_num = outputs['feat_num']
        epoch = outputs['epoch']

        layers = np.shape(bit_level)[0]
        total_num = 0
        total_num_a = 0

        total_bit_num = 0
        total_bita_num = 0

        real_bit_num = 0
        real_bita_num = 0

        total_num_cr = 0
        total_bit_num_cr = 0

        total_layer = 0
        bit_loss = 0

        for i in range(layers):
            total_num += param_num[i]
            total_bit_num += (param_num[i] * bit_level[i])
            real_bit_num += (param_num[i] * (bit_level[i].round()))

            total_num_a += param_num[i] * feat_num[i]
            total_bita_num += (param_num[i] * feat_num[i] * bit_level[i] * bita_level[i])
            real_bita_num += (param_num[i] * feat_num[i] * bit_level[i].round() * bita_level[i].round())

            total_layer += 1
            bit_loss = bit_loss + ((torch.abs(bit_level[i] - bit_level[i].round()) + torch.abs(bita_level[i] - bita_level[i].round()))/2)


        pred_bit = total_bit_num / total_num
        pred_bit_real = real_bit_num / total_num

        pred_bita = total_bita_num / total_num_a
        pred_bita_real = real_bita_num / total_num_a

        bit_loss = 0.1 * (epoch / 400) * (bit_loss / total_layer)

        # if pred_bit <= target_bit:
        #     t_reg_loss = torch.tensor(0.0)
        # else:
        #     t_reg_loss = l1_fn(pred_bit, target_bit)

        # t_reg_loss = (0.1*l1_fn(pred_bit, target_bit)) + (0.02*l1_fn(pred_bita, target_bita))
        t_reg_loss = 0.02*l1_fn(pred_bita, target_bita)

        loss_dict['loss'] = gt_loss + t_reg_loss +  bit_loss
        loss_dict['t_reg_loss'] = t_reg_loss
        loss_dict['bit_loss'] = bit_loss
        loss_dict['gt_loss'] = gt_loss
        loss_dict['pred_bit'] = pred_bit_real
        loss_dict['pred_bita'] = pred_bita_real

        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def cross_entropy_t_reg_l1_cr_reg_10(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)
    l2_fn = torch.nn.MSELoss(reduction=reduction)

    def loss_fn(outputs, labels, **_):
        loss_dict = dict()
        cr_dict = dict()
        pred = outputs['out']
        gt_loss = cross_entropy_fn(pred, labels)

        target_bit = torch.tensor(2.8).cuda()
        target_bita = torch.tensor(2.8).cuda()
        reg_bit = torch.tensor(1).cuda()
        bit_level = outputs['bit_level'] # integer bit-width
        param_num = outputs['param_num']
        bita_level = outputs['bita_level']
        feat_num = outputs['feat_num']
        epoch = outputs['epoch']

        layers = np.shape(bit_level)[0]
        total_num = 0
        total_num_a = 0

        total_bit_num = 0
        total_bita_num = 0

        real_bit_num = 0
        real_bita_num = 0

        total_num_cr = 0
        total_bit_num_cr = 0

        total_layer = 0
        bit_loss = 0

        for i in range(layers):
            total_num += param_num[i]
            total_bit_num += (param_num[i] * bit_level[i])
            real_bit_num += (param_num[i] * (bit_level[i].round()))

            total_num_a += feat_num[i]
            total_bita_num += (feat_num[i] * bita_level[i])
            real_bita_num += (feat_num[i] * bita_level[i].round())

            total_layer += 1
            bit_loss = bit_loss + ((torch.abs(bit_level[i] - bit_level[i].round()) + torch.abs(bita_level[i] - bita_level[i].round()))/2)


        pred_bit = total_bit_num / total_num
        pred_bit_real = real_bit_num / total_num

        pred_bita = total_bita_num / total_num_a
        pred_bita_real = real_bita_num / total_num_a

        bit_loss = 0.0 * (epoch / 500500) * (bit_loss / total_layer)

        # if pred_bit <= target_bit:
        #     t_reg_loss = torch.tensor(0.0)
        # else:
        #     t_reg_loss = l1_fn(pred_bit, target_bit)

        t_reg_loss = (0.1*l1_fn(pred_bit, target_bit)) + (0.1*l1_fn(pred_bita, target_bita))
        # t_reg_loss = 0.03*l1_fn(pred_bit, target_bit)

        loss_dict['loss'] = gt_loss + t_reg_loss +  bit_loss
        loss_dict['t_reg_loss'] = t_reg_loss
        loss_dict['bit_loss'] = bit_loss
        loss_dict['gt_loss'] = gt_loss
        loss_dict['pred_bit'] = pred_bit_real
        loss_dict['pred_bita'] = pred_bita_real

        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def cross_entropy_t_reg_l1_cr_reg_10_reg(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)
    l2_fn = torch.nn.MSELoss(reduction=reduction)

    def loss_fn(outputs, labels, **_):
        loss_dict = dict()
        cr_dict = dict()
        pred = outputs['out']
        gt_loss = cross_entropy_fn(pred, labels)

        target_bit = torch.tensor(1.95).cuda()
        target_bita = torch.tensor(2.024).cuda()
        reg_bit = torch.tensor(1).cuda()
        bit_level = outputs['bit_level'] # integer bit-width
        param_num = outputs['param_num']
        bita_level = outputs['bita_level']
        feat_num = outputs['feat_num']
        epoch = outputs['epoch']
        bit_w_score = outputs['bit_w_score']
        bit_a_score = outputs['bit_a_score']

        layers = np.shape(bit_level)[0]
        total_num = 0
        total_num_a = 0

        total_bit_num = 0
        total_bita_num = 0

        real_bit_num = 0
        real_bita_num = 0

        total_num_cr = 0
        total_bit_num_cr = 0

        total_layer = 0
        bit_loss = 0

        for i in range(layers):
            total_num += param_num[i]
            total_bit_num += (param_num[i] * RoundWithGradient.apply(bit_level[i]))
            # total_bit_num += (param_num[i] * bit_level[i]) * (1-bit_w_score[i-1])
            # total_bit_num += (param_num[i] * bit_level[i])
            real_bit_num += (param_num[i] * (bit_level[i].round()))

            total_num_a += feat_num[i]
            total_bita_num += (feat_num[i] * RoundWithGradient.apply(bita_level[i]))
            # total_bita_num += (feat_num[i] * bita_level[i]) * (1-bit_a_score[i-1])
            # total_bita_num += (feat_num[i] * bita_level[i])
            real_bita_num += (feat_num[i] * bita_level[i].round())

            total_layer += 1
            bit_loss = bit_loss + ((torch.abs(bit_level[i] - (bit_level[i].floor() + 0.1)) + torch.abs(bita_level[i] - (bita_level[i].floor() + 0.1))))
            # bit_loss = bit_loss + ((torch.abs(bit_level[i] - bit_level[i].floor()) + torch.abs(bita_level[i] - bita_level[i].floor())))


        pred_bit = total_bit_num / total_num
        pred_bit_real = real_bit_num / total_num

        pred_bita = total_bita_num / total_num_a
        pred_bita_real = real_bita_num / total_num_a

        bit_loss = 0.5 * ((epoch / 135150) ** 1) * (bit_loss / total_layer)

        # if pred_bit <= target_bit:
        #     t_reg_loss = torch.tensor(0.0)
        # else:
        #     t_reg_loss = l1_fn(pred_bit, target_bit)
        # weight = (epoch / 135150)

        # t_reg_loss = (0.5*l1_fn(pred_bit, target_bit)) + (0.5*l1_fn(pred_bita, target_bita))
        # t_reg_loss = (1 - (epoch / 135150)) * t_reg_loss

        # t_reg_loss = 0.5*(torch.clamp(pred_bit - target_bit, min=0.0)) + 0.5*(torch.clamp(pred_bita - target_bita, min=0.0))
        t_reg_loss = (torch.clamp(pred_bit - target_bit, min=0.0)) + (torch.clamp(pred_bita - target_bita, min=0.0))

        # t_reg_loss = 0.03*l1_fn(pred_bit, target_bit)

        loss_dict['loss'] = gt_loss + t_reg_loss
        loss_dict['t_reg_loss'] = t_reg_loss
        loss_dict['bit_loss'] = bit_loss
        loss_dict['gt_loss'] = gt_loss
        loss_dict['pred_bit'] = pred_bit_real
        loss_dict['pred_bita'] = pred_bita_real

        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def regularization(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)

    def loss_fn(outputs, labels, reg_factors, **_):
        loss_dict = dict()
        gt_loss = cross_entropy_fn(outputs, labels)
        reg_loss = 0
        for i in range(len(reg_factors)):
            reg_loss += torch.mean((torch.pow(reg_factors[i]-1, 2)*torch.pow(reg_factors[i]+1, 2)))
        reg_loss = reg_loss / len(reg_factors)
        loss_dict['loss'] = gt_loss + reg_loss
        loss_dict['gt_loss'] = gt_loss
        loss_dict['reg_loss'] = reg_loss
        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def regularization_temp(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)

    def loss_fn(outputs, labels, reg_factors, **_):
        loss_dict = dict()
        gt_loss = cross_entropy_fn(outputs, labels)
        reg_loss = 0
        for i in range(len(reg_factors)):
            reg_loss += torch.mean((torch.pow(reg_factors[i]-1, 2)*torch.pow(reg_factors[i]+1, 2)))
        reg_loss = reg_loss / len(reg_factors)
        loss_dict['loss'] = gt_loss + reg_loss
        loss_dict['gt_loss'] = gt_loss
        loss_dict['reg_loss'] = reg_loss
        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}


def get_loss(config):
    f = globals().get(config.loss.name)
    return f(**config.loss.params)
