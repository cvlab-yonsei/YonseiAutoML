import os
import tqdm
import argparse
import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np
import skimage
import os
import glob
from skimage.io import imread
import skimage
import math
import time
# from datasets import get_train_dataloader, get_valid_dataloader, get_test_dataloader
from models import get_model
from losses import get_loss
from optimizers import get_optimizer, get_q_optimizer
from schedulers import get_scheduler, get_q_scheduler
# from visualizers import get_visualizer
from tensorboardX import SummaryWriter
from evaluators import accuracy

import utils.config
import utils.checkpoint
from utils import AverageMeter

from torch.utils.data import DataLoader
import torchvision

from models.modules import DSQConv_a, DSQConv_8bit, DSQLinear

from torch.utils.data.sampler import SubsetRandomSampler


device = None
init = True

total_large_quant_error = 0
total_layer = 0
max_ratio_constant_global = 1e-3

total_grad_num = 0

def train_single_epoch(config, student_model, dataloader, criterion,
                       optimizer, q_optimizer, epoch, writer, postfix_dict):
    student_model.train()
    batch_size = config.train.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)
    global init 

    log_dict = {}

    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)

    for m in student_model.modules():
        if isinstance(m, DSQConv_a):
            m.training_ratio = (epoch / 400)
            m.quant_w = True
            m.quant_a = True

    for i, (imgs, labels) in tbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        q_optimizer.zero_grad()
        # b_optimizer.zero_grad()


        pred_dict = student_model(imgs)
        loss = criterion['train'](pred_dict, labels)

        for k, v in loss.items():
            log_dict[k] = v.item()

        loss['loss'].backward()
        if init == False:
            optimizer.step()
            q_optimizer.step()
        # if epoch < 200:
        #     b_optimizer.step()

        ## logging
        f_epoch = epoch + i / total_step
        log_dict['lr'] = optimizer.param_groups[0]['lr']
        for key, value in log_dict.items():
            postfix_dict['train/{}'.format(key)] = value

        desc = '{:5s}'.format('train')
        desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
        tbar.set_description(desc)
        tbar.set_postfix(**postfix_dict)

        iter_ratio = (i+1) / total_step

        update_interval(student_model, epoch, iter_ratio)

        init = False

        global total_grad_num
        # print(total_grad_num)
        total_grad_num = 0

        # tensorboard
        if i % 10 == 0:

            log_step = int(f_epoch * 1280)
            if writer is not None:
                for key, value in log_dict.items():
                    writer.add_scalar('train/{}'.format(key), value, log_step)
                    
                q_log_dict = log_q_param_train(student_model)

                for key, value in q_log_dict[0].items():
                    if writer is not None:
                        writer.add_scalar('{}'.format(key), value, log_step)

        # if i % 30 == 0:
        #     log_step = int(f_epoch * 1280)
        #     if writer is not None:
        #         log_q_fig_train(student_model, writer, iteration=log_step)

def update_interval(model, epoch, iter_ratio):

    global init
    global total_grad_num
    moment = 0.999
    constant = 1e-1

    for m in model.modules():
        # if isinstance(m, DSQConv_a) or isinstance(m, DSQConv_8bit) or isinstance(m, DSQLinear)
        if isinstance(m, DSQConv_a):
            output_grad = m.output_buff.grad
            output_grad_quant = m.quant_output_buff.grad
            clip_ratio = m.clip_ratio.clone()
            moment_ratio = m.moment_ratio.clone()
            max_ratio_constant = m.max_ratio_constant
            max_ratio_constant_q = m.max_ratio_constant_q

            # output_grad_scale = m.output_grad_scale
            # quant_output_grad_scale = m.quant_output_grad_scale
            
            weight = m.weight.clone()
            sW = m.sW.clone()

            mean_ratio = m.mean_ratio.clone()
            std_ratio = m.std_ratio.clone()
            grad_max = m.grad_max.clone()
            layer_num = m.layer_num

            bit = m.bit
            alpha = m.alpha
            
            threshold = 0
            max_thres = 0

            moment_max_ratio = m.max_ratio

            kurtosis = ((output_grad - output_grad.mean()) ** 4).mean() / (output_grad.std() ** 4)
            maxtosis = ((output_grad) ** 4).mean() / (output_grad.max() ** 4)
            maxtosis_half = ((output_grad - output_grad.mean()) ** 4).mean() / ((3 * output_grad.std()) ** 4)

            grad_large_ratio_3 = (output_grad.abs() > output_grad.abs().max() * 0.3).float().mean()

            total_grad_num = total_grad_num + (output_grad == output_grad).float().sum()

        
            # max_ratio_constant = grad_large_ratio_9 * 200

            # max_ratio_constant = kurtosis * grad_large_ratio_3 * 0.5e-2

            b, _, _, _ = output_grad.size()
            max_mean = output_grad.abs().view(b, -1).max(dim=1)[0].mean()
            # max_ratio_constant = (output_grad.abs() > 0.9 * max_mean).float().mean()

            grad_max = output_grad.abs().max()
            # grad_max = (moment * grad_max) + ((1-moment) * output_grad.abs().max())

            if init:
                # output_grad_scale = output_grad.abs().mean()
                # quant_output_grad_scale = output_grad_quant.abs().mean()
                max_ratio_constant = 1e-3
                grad_max = output_grad.abs().max()
                mean_ratio = output_grad.mean()
                std_ratio = output_grad.std()

                moment_ratio = torch.tensor(0.0)
                clip_ratio = torch.tensor(1.0)

                MSR = grad_max / std_ratio
                KMR = kurtosis / MSR
                KMR4 = kurtosis / (MSR ** 4)
                KMR4 = torch.pow(KMR4, 0.5)
                # max_ratio_constant = constant * KMR4

                # decay = torch.exp(-MSR / 10)
                # max_ratio_constant = 1e-3 * decay * constant
                # max_ratio_constant = torch.clamp(std_ratio * 50, max=0.999).item() 
                # max_ratio_constant = (output_grad.abs() >= grad_max * 0.45).float().mean()

                if max_ratio_constant >= 0.999:
                    moment_ratio = torch.tensor(0.999)
                    clip_ratio = torch.tensor(0.001)
                else:
                    while threshold <= 0:
                        output_grad_clip_ratio = (output_grad.abs() > grad_max * clip_ratio).float().mean()
                        moment_ratio = (moment * moment_ratio) + ((1-moment) * output_grad_clip_ratio)
                        threshold = (moment_ratio - ((1 / ((2**bit) - 1)) * max_ratio_constant))
                        threshold = torch.sign(threshold)
                        clip_ratio = clip_ratio + ((1-moment) * threshold)
                
                alpha = torch.tensor(1.0)
                moment_ratio = torch.tensor(0.0)
                
                if max_ratio_constant >= 0.999:
                    moment_ratio = torch.tensor(0.999)
                    alpha = torch.tensor(0.001)
                else:
                    while max_thres <= 0:
                        output_grad_alpha_ratio = (output_grad.abs() > grad_max * alpha).float().mean()
                        moment_ratio = (moment * moment_ratio) + ((1-moment) * output_grad_alpha_ratio)
                        max_thres = (moment_ratio - max_ratio_constant)
                        max_thres = torch.sign(max_thres)
                        alpha = alpha + ((1-moment) * max_thres)

                m.clip_ratio.data.fill_(torch.clamp(clip_ratio, min=0.001, max=1))
                m.alpha = torch.clamp(alpha, min=0.001, max=1)
                m.std_ratio = std_ratio
                m.mean_ratio = mean_ratio
                m.grad_max = grad_max
                m.max_ratio_constant = max_ratio_constant
                # m.output_grad_scale = output_grad_scale
                # m.quant_output_grad_scale = quant_output_grad_scale
            else:
                if grad_max != 0:

                    # global max_ratio_constant_global

                    # if sW >= weight.std() * 4 and clip_ratio <= 0.99:
                    #     max_ratio_constant = torch.clamp(torch.tensor(max_ratio_constant) - (1e-5), min=1e-5).item()


                    # output_grad_scale = (moment * output_grad_scale) + (1-moment) * output_grad.abs().mean()
                    # quant_output_grad_scale = (moment * quant_output_grad_scale) + (1-moment) * output_grad_quant.abs().mean()

                    # max_update = torch.sign(quant_output_grad_scale - output_grad_scale)

                    # max_ratio_constant = max_ratio_constant + (5e-6 * max_update)
                    
                    # if sW >= weight.std() * 4 and clip_ratio <= 0.99:
                    #     max_ratio_constant = torch.clamp(torch.tensor(max_ratio_constant) - (1e-5), min=1e-5).item()

                    # if max_ratio_constant != max_ratio_constant_global:
                    #     max_ratio_constant = max_ratio_constant_global
                        

                    mean_ratio = (moment * mean_ratio) + ((1-moment) * (output_grad.mean()))
                    std_ratio = output_grad.std()
                    max_ratio_constant_q = (output_grad_quant.abs() > grad_max * clip_ratio).float().mean()

                    MSR = grad_max / std_ratio
                    KMR = kurtosis / MSR
                    KMR4 = kurtosis / (MSR ** 4)
                    KMR4 = torch.pow(KMR4, 0.5)
                    # max_ratio_constant = constant * KMR4

                    # decay = torch.exp(-MSR / 10)
                    # max_ratio_constant = (output_grad.abs() >= grad_max * 0.45).float().mean()
                    # max_ratio_constant = 1e-3 * decay * constant
                    # max_ratio_constant = torch.clamp(std_ratio * 50, max=0.999).item() 
                    
                    output_grad_clip_ratio = (output_grad.abs() > grad_max * clip_ratio).float().mean()
                    moment_ratio = output_grad_clip_ratio
                    threshold = (moment_ratio - ((1 / ((2**bit) - 1)) * max_ratio_constant))
                    threshold = torch.sign(threshold)
                    clip_ratio = clip_ratio + ((1-moment) * threshold)

                    output_grad_alpha_ratio = (output_grad.abs() > grad_max * alpha).float().mean()
                    max_thres = torch.sign(output_grad_alpha_ratio - max_ratio_constant)
                    alpha = alpha + ((1-moment) * max_thres)

                    m.clip_ratio.data.fill_(torch.clamp(clip_ratio, min=0.001, max=1))
                    m.alpha = torch.clamp(alpha, min=0.001, max=1)
                    m.std_ratio = std_ratio
                    m.mean_ratio = mean_ratio
                    m.grad_max = grad_max
                    m.max_ratio_constant = max_ratio_constant
                    m.sW.data.fill_(sW)
                    # m.output_grad_scale = output_grad_scale
                    # m.quant_output_grad_scale = quant_output_grad_scale

def log_q_fig_train(model, writer, index=0, iteration=0):

    for m in model._modules:
        if len(model._modules[m]._modules) > 0:
            index = log_q_fig_train(model._modules[m], writer, index, iteration)

        else:
            if hasattr(model._modules[m], 'init'):
                # fig = plt.figure()
                # quant_output_grad = model._modules[m].quant_output_buff.grad
                # plt.hist(quant_output_grad.detach().cpu().numpy().reshape(-1), bins=100)
                # writer.add_figure("{}_{}/q_grad_fig".format(m, index), fig, iteration)

                # fig = plt.figure()
                # output_grad = model._modules[m].output_buff.grad
                # plt.hist(output_grad.detach().cpu().numpy().reshape(-1), bins=100)
                # writer.add_figure("{}_{}/grad_fig".format(m, index), fig, iteration)

                fig = plt.figure()
                fp_act = model._modules[m].fp_act
                plt.hist(fp_act.detach().cpu().numpy().reshape(-1), bins=100)
                writer.add_figure("{}_{}/act_fp".format(m, index), fig, iteration)

                fig = plt.figure()
                q_act = model._modules[m].q_act
                plt.hist(q_act.detach().cpu().numpy().reshape(-1), bins=100)
                writer.add_figure("{}_{}/act_q".format(m, index), fig, iteration)

                fig = plt.figure()
                fp_weight = model._modules[m].fp_weight
                plt.hist(fp_weight.detach().cpu().numpy().reshape(-1), bins=100)
                writer.add_figure("{}_{}/weight_fp".format(m, index), fig, iteration)

                fig = plt.figure()
                q_weight = model._modules[m].q_weight
                plt.hist(q_weight.detach().cpu().numpy().reshape(-1), bins=100)
                writer.add_figure("{}_{}/weight_q".format(m, index), fig, iteration)

            index = index + 1

    return index

def quantize(gradients, bit, clipping_ratio):

    ratio = clipping_ratio

    max_value = float(gradients.abs().max()) * ratio
    min_value = -max_value

    if min_value == 0:
        min_value = min_value + 1e-8
        max_value = -min_value
    
    qmin = 0.
    qmax = 2. ** bit - 2.

    scale = (max_value - min_value) / (qmax - qmin)

    grad_out = (gradients - min_value) / scale
    grad_out = grad_out + qmin
    grad_out = torch.clamp(grad_out, min=qmin, max=qmax)
    grad_out = torch.round(grad_out)
    grad_out = ((grad_out - qmin) * scale) + min_value

    return grad_out

def log_q_param_train(model, log_dict={}, index=0):

    for m in model._modules:
        if len(model._modules[m]._modules) > 0:
            log_dict, index = log_q_param_train(model._modules[m], log_dict, index)

        else:
            if hasattr(model._modules[m], 'init'):
                if hasattr(model._modules[m], 'sB'):
                    var = list(model._modules[m].parameters())[2:]
                else:
                    var = list(model._modules[m].parameters())[1:]

                log_dict['{}_{}/sW'.format(m, index-1)] = var[0].item()
                if hasattr(model._modules[m], 'uA'):
                    log_dict['{}_{}/uA'.format(m, index-1)] = var[1].item()
                # log_dict['{}_{}/beta'.format(m, index-1)] = var[1].item()
                # log_dict['{}_{}/uA'.format(m, index-1)] = var[2].item()

                clip_ratio = model._modules[m].clip_ratio.clone()
                bit = model._modules[m].bit.clone()
                alpha = model._modules[m].alpha.clone()
                mean_ratio = model._modules[m].mean_ratio.clone()
                std_ratio = model._modules[m].std_ratio.clone()

                sW_grad = model._modules[m].sW.grad
                uA_grad = model._modules[m].uA.grad

                sW_grad_scale = sW_grad
                uA_grad_scale = uA_grad
                # transition_ratio = model._modules[m].weight_diff.clone()

                quant_output_grad = model._modules[m].quant_output_buff.grad
                output_grad = model._modules[m].output_buff.grad

                output_grad_scale = output_grad.abs().mean()
                quant_output_grad_scale = quant_output_grad.abs().mean()

                ratio_scale = quant_output_grad_scale / output_grad_scale

                kurtosis = ((output_grad - output_grad.mean()) ** 4).mean() / (output_grad.std() ** 4)
                maxtosis = ((output_grad) ** 4).mean() / (output_grad.abs().max() ** 4)

                output_grad_norm = output_grad / output_grad.abs().max()

                ori_std = output_grad_norm.std()
                large_std = output_grad_norm[output_grad.abs() > output_grad.abs().max() * 0.1].std()

                # grad_large = output_grad[output_grad.abs() > output_grad.abs().max() * 0.1]
                # grad_large_magnitude_1 = (((grad_large / grad_large.abs().max()).abs()) ** 4).mean()

                # grad_large = output_grad[output_grad.abs() > output_grad.abs().max() * 0.2]
                # grad_large_magnitude_2 = (((grad_large / grad_large.abs().max()).abs()) ** 4).mean()

                # grad_large = output_grad[output_grad.abs() > output_grad.abs().max() * 0.3]
                # grad_large_magnitude_3 = (((grad_large / grad_large.abs().max()).abs()) ** 4).mean()

                # grad_large = output_grad[output_grad.abs() > output_grad.abs().max() * 0.5]
                # grad_large_magnitude_5 = (((grad_large / grad_large.abs().max()).abs()) ** 4).mean()

                # grad_large_ratio_1 = (output_grad.abs() > output_grad.abs().max() * 0.1).float().mean()
                # grad_large_ratio_2 = (output_grad.abs() > output_grad.abs().max() * 0.2).float().mean()
                # grad_large_ratio_3 = (output_grad.abs() > output_grad.abs().max() * 0.3).float().mean()
                # grad_large_ratio_4 = (output_grad.abs() > output_grad.abs().max() * 0.4).float().mean()
                # grad_large_ratio_5 = (output_grad.abs() > output_grad.abs().max() * 0.5).float().mean()

                # grad_large_ratio_8 = (output_grad.abs() > output_grad.abs().max() * 0.8).float().mean()
                # grad_large_ratio_9 = (output_grad.abs() > output_grad.abs().max() * 0.9).float().mean()

                # large_ratio_2 = grad_large_ratio_2 / grad_large_ratio_1
                # large_ratio_3 = grad_large_ratio_3 / grad_large_ratio_1
                # large_ratio_4 = grad_large_ratio_4 / grad_large_ratio_1
                # large_ratio_5 = grad_large_ratio_5 / grad_large_ratio_1
                # large_ratio_8 = grad_large_ratio_8 / grad_large_ratio_1
                # large_ratio_9 = grad_large_ratio_9 / grad_large_ratio_1

                # large_ratio_ori = grad_large_ratio_9 * 200

                kurtosis_q = ((quant_output_grad - quant_output_grad.mean()) ** 4).mean() / (quant_output_grad.std() ** 4)
                kurtosis_q_var = ((quant_output_grad - output_grad.mean()) ** 4).mean() / (output_grad.std() ** 4)

                kurtosis_ratio = kurtosis_q / kurtosis
                kurtosis_ratio_var = kurtosis_q_var / kurtosis

                grad_max = output_grad.abs().max()
                max_std_ratio = grad_max / std_ratio

                KMR = kurtosis / max_std_ratio
                KMR2 = kurtosis / (max_std_ratio ** 2)
                KMR3 = kurtosis / (max_std_ratio ** 3)
                KMR4 = kurtosis / (max_std_ratio ** 4)
                # KMR4 = torch.pow(KMR4, 0.5)

                max_ratio_constant = (output_grad.abs() > grad_max * alpha).float().mean()
                max_ratio_constant_q = (quant_output_grad.abs() > grad_max * alpha).float().mean()

                q_grad_max = quant_output_grad.abs().max()
                q_std_ratio = quant_output_grad.std()
                q_max_std_ratio = q_grad_max / (q_std_ratio)


                quant_error_matrix = (quant_output_grad - output_grad).abs() / grad_max
                quant_error = quant_error_matrix.mean()

                quant_output_ours = quantize(output_grad, bit, clip_ratio)
                quant_error_matrix_ours = (quant_output_ours - output_grad).abs() / grad_max
                quant_error_ours = quant_error_matrix_ours.mean()

                large_grad = (output_grad.abs() >= grad_max * alpha)
                quant_error_large = quant_error_matrix[large_grad].mean()
                quant_error_large_ours = quant_error_matrix_ours[large_grad].mean() 

                large_ratio = maxtosis * kurtosis * 1e-1

                wegith_grad = model._modules[m].weight.grad.clone()
                grad_scale_weight = ((wegith_grad ** 2).sum() ** 0.5)

                log_dict['{}_{}/q_error_grad'.format(m, index-1)] = quant_error.item()
                log_dict['{}_{}/q_error_grad_large'.format(m, index-1)] = quant_error_large.item()
                log_dict['{}_{}/q_error_grad_large_ours'.format(m, index-1)] = quant_error_large_ours.item()
                log_dict['{}_{}/ratio_clip'.format(m, index-1)] = clip_ratio.item()
                log_dict['{}_{}/ratio_alpha'.format(m, index-1)] = alpha.item()

                log_dict['{}_{}/ratio_mean'.format(m, index-1)] = mean_ratio.item()
                log_dict['{}_{}/ratio_std'.format(m, index-1)] = std_ratio.item()
                log_dict['{}_{}/ratio_std_q'.format(m, index-1)] = q_std_ratio.item()

                log_dict['{}_{}/ratio_scale'.format(m, index-1)] = ratio_scale.item()

                log_dict['{}_{}/ratio_max_to_std'.format(m, index-1)] = max_std_ratio.item()

                log_dict['{}_{}/ratio_max_constant'.format(m, index-1)] = max_ratio_constant.item()

                log_dict['{}_{}/a_grad_scale'.format(m, index-1)] = output_grad_scale.item()
                log_dict['{}_{}/a_grad_scale_q'.format(m, index-1)] = quant_output_grad_scale.item()

                # log_dict['{}_{}/ratio_large'.format(m, index-1)] = large_ratio_ori.item()
                # log_dict['{}_{}/ratio_large_pow2'.format(m, index-1)] = (large_ratio ** 2).item()

                # log_dict['{}_{}/ratio_large_1'.format(m, index-1)] = large_ratio_1.item()
                # log_dict['{}_{}/ratio_large_2'.format(m, index-1)] = large_ratio_2.item()
                # log_dict['{}_{}/ratio_large_3'.format(m, index-1)] = large_ratio_3.item()

                log_dict['{}_{}/a_grad_scale_weight'.format(m, index-1)] = grad_scale_weight.item()
                log_dict['{}_{}/a_grad_scale_uA'.format(m, index-1)] = uA_grad_scale.item()
                log_dict['{}_{}/a_grad_scale_sW'.format(m, index-1)] = sW_grad_scale.item()

                log_dict['{}_{}/std_ori'.format(m, index-1)] = ori_std.item()
                log_dict['{}_{}/std_large'.format(m, index-1)] = large_std.item()

                # log_dict['{}_{}/large_grad_mag_0.1'.format(m, index-1)] = grad_large_magnitude_1.item()
                # log_dict['{}_{}/large_grad_mag_0.2'.format(m, index-1)] = grad_large_magnitude_2.item()
                # log_dict['{}_{}/large_grad_mag_0.3'.format(m, index-1)] = grad_large_magnitude_3.item()
                # log_dict['{}_{}/large_grad_mag_0.5'.format(m, index-1)] = grad_large_magnitude_5.item()

                # log_dict['{}_{}/large_grad_ratio_0.1'.format(m, index-1)] = grad_large_ratio_1.item()
                # log_dict['{}_{}/large_grad_ratio_0.2'.format(m, index-1)] = grad_large_ratio_2.item()
                # log_dict['{}_{}/large_grad_ratio_0.3'.format(m, index-1)] = grad_large_ratio_3.item()
                # log_dict['{}_{}/large_grad_ratio_0.4'.format(m, index-1)] = grad_large_ratio_4.item()
                # log_dict['{}_{}/large_grad_ratio_0.5'.format(m, index-1)] = grad_large_ratio_5.item()
                # log_dict['{}_{}/large_grad_ratio_0.8'.format(m, index-1)] = grad_large_ratio_8.item()
                # log_dict['{}_{}/large_grad_ratio_0.9'.format(m, index-1)] = grad_large_ratio_9.item()

                # log_dict['{}_{}/large_grad_ratio_0.1_our'.format(m, index-1)] = large_ratio_1.item()
                # log_dict['{}_{}/large_grad_ratio_0.2_our'.format(m, index-1)] = large_ratio_2.item()
                # log_dict['{}_{}/large_grad_ratio_0.3_our'.format(m, index-1)] = large_ratio_3.item()

                # log_dict['{}_{}/ratio_0.9_to_0.1'.format(m, index-1)] = large_ratio_9.item()

                

                # log_dict['{}_{}/ratio_3std_large'.format(m, index-1)] = valid_grad_ratio_3.item()
                # log_dict['{}_{}/ratio_5std_large'.format(m, index-1)] = valid_grad_ratio_5.item()
                # log_dict['{}_{}/ratio_7std_large'.format(m, index-1)] = valid_grad_ratio_7.item()
                # log_dict['{}_{}/ratio_9std_large'.format(m, index-1)] = valid_grad_ratio_9.item()
                # log_dict['{}_{}/ratio_zero_grad'.format(m, index-1)] = zero_grad_ratio.item()
                # log_dict['{}_{}/ratio_max_constant_q'.format(m, index-1)] = max_ratio_constant_q.item()
                
                log_dict['{}_{}/kurtosis'.format(m, index-1)] = kurtosis.item()
                log_dict['{}_{}/kurtosis_q'.format(m, index-1)] = kurtosis_q.item()
                log_dict['{}_{}/kurtosis_ratio'.format(m, index-1)] = kurtosis_ratio.item()

                log_dict['{}_{}/kurtosis_q_var'.format(m, index-1)] = kurtosis_q_var.item()
                log_dict['{}_{}/kurtosis_ratio_var'.format(m, index-1)] = kurtosis_ratio_var.item()
                # log_dict['{}_{}/kurtosis_max'.format(m, index-1)] = maxtosis.item()
                # log_dict['{}_{}/kurtosis_half'.format(m, index-1)] = maxtosis_half.item()
                # log_dict['{}_{}/kurtosis_to_max'.format(m, index-1)] = KMR.item()
                # log_dict['{}_{}/kurtosis_to_max_pow2'.format(m, index-1)] = KMR2.item()
                # log_dict['{}_{}/kurtosis_to_max_pow3'.format(m, index-1)] = KMR3.item()
                # log_dict['{}_{}/kurtosis_to_max_pow4'.format(m, index-1)] = KMR4.item()
                # log_dict['{}_{}/kurtosis_q'.format(m, index-1)] = kurtosis_q.item()

                # log_dict['{}_{}/ratio_max_to_std_q'.format(m, index-1)] = q_max_std_ratio.item()
                # log_dict['{}_{}/ratio_transition'.format(m, index-1)] = transition_ratio.item()

            elif hasattr(model._modules[m], 'affine'):
                gamma = model._modules[m].weight.clone()
                beta = model._modules[m].bias.clone()

                gamma_grad = model._modules[m].weight.grad.clone()
                beta_grad = model._modules[m].bias.grad.clone()

                gamma_mag = ((gamma ** 2).sum() ** 0.5)
                beta_mag = ((beta ** 2).sum() ** 0.5)

                gamma_grad_mag = ((gamma_grad ** 2).sum() ** 0.5)
                beta_grad_mag = ((beta_grad ** 2).sum() ** 0.5)

                log_dict['{}_{}/gamma_mag'.format(m, index-1)] = gamma_mag.item()
                log_dict['{}_{}/beta_mag'.format(m, index-1)] = beta_mag.item()

                log_dict['{}_{}/gamma_grad_mag'.format(m, index-1)] = gamma_grad_mag.item()
                log_dict['{}_{}/beta_grad_mag'.format(m, index-1)] = beta_grad_mag.item()
                
            index = index + 1

    return log_dict, index

def log_q_param(model, log_dict={}, index=0):

    for m in model._modules:
        if len(model._modules[m]._modules) > 0:
            log_dict, index = log_q_param(model._modules[m], log_dict, index)

        else:
            if hasattr(model._modules[m], 'init'):
                if hasattr(model._modules[m], 'sB'):
                    var = list(model._modules[m].parameters())[2:]
                else:
                    var = list(model._modules[m].parameters())[1:]
                log_dict['{}_{}/sW'.format(m, index)] = var[0].item()
                if hasattr(model._modules[m], 'uA'):
                    log_dict['{}_{}/uA'.format(m, index)] = var[1].item()
                # log_dict['{}_{}/beta'.format(m, index)] = var[1].item()
                # log_dict['{}_{}/uA'.format(m, index)] = var[2].item()

            index = index + 1

    return log_dict, index

def evaluate_single_epoch(config, student_model,
                          dataloader, criterion, epoch, writer,
                          postfix_dict, eval_type):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    q_losses = AverageMeter()
    q_top1 = AverageMeter()
    q_top5 = AverageMeter()
    student_model.eval()

    with torch.no_grad():
        batch_size = config.eval.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)

        for m in student_model.modules():
            if isinstance(m, DSQConv_a):
                m.training_ratio = (epoch / 400)
                m.quant_w = True
                m.quant_a = True

        for i, (imgs, labels) in tbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            pred_dict = student_model(imgs)
            train_loss = criterion['val'](pred_dict['out'], labels)
            prec1, prec5 = accuracy(pred_dict['out'].data, labels.data, topk=(1,5))
            prec1 = prec1[0]
            prec5 = prec5[0]

            losses.update(train_loss.item(), labels.size(0))
            top1.update(prec1, labels.size(0))
            top5.update(prec5, labels.size(0))

            ## Logging
            f_epoch = epoch + i / total_step
            desc = '{:5s}'.format(eval_type)
            desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
            tbar.set_description(desc)
            tbar.set_postfix(**postfix_dict)

        ## logging
        log_dict = {}
        log_dict['loss'] = losses.avg
        log_dict['top1'] = top1.avg.item()
        log_dict['top5'] = top5.avg.item()

        for key, value in log_dict.items():
            if writer is not None:
                writer.add_scalar('{}/{}'.format(eval_type, key), value, epoch)
            postfix_dict['{}/{}'.format(eval_type, key)] = value

        q_log_dict = log_q_param(student_model)

        for key, value in q_log_dict[0].items():
            if writer is not None:
                writer.add_scalar('{}'.format(key), value, epoch)

        return log_dict['top1'], log_dict['top5']

def train(config, student_model, dataloaders, criterion,
          optimizer, q_optimizer, scheduler, q_scheduler, writer, start_epoch):
    num_epochs = config.train.num_epochs
    if torch.cuda.device_count() > 1:
        student_model = torch.nn.DataParallel(student_model)

    postfix_dict = {'train/lr': 0.0,
                    'train/loss': 0.0,
                    'train/accuracy': 0.0,
                    'test/accuracy':0.0,
                    'test/loss':0.0}

    best_accuracy = 0.0

    for epoch in range(start_epoch, num_epochs):
        # train phase
        train_single_epoch(config, student_model, dataloaders['train'],
                           criterion, optimizer, q_optimizer, epoch, writer,
                           postfix_dict)

        # test phase
        top1, top5 = evaluate_single_epoch(config, student_model,
                                           dataloaders['test'],
                                           criterion, epoch, writer,
                                           postfix_dict, eval_type='test')

        scheduler.step()
        q_scheduler.step()

        utils.checkpoint.save_checkpoint(config, student_model, optimizer, scheduler, q_optimizer,
                                         q_scheduler, None, None, epoch, 0, 'student')

        if best_accuracy < top1:
            best_accuracy = top1

    return {'best_accuracy': best_accuracy}

def qparam_extract(model):

    var = list()

    for m in model._modules:
        if len(model._modules[m]._modules) > 0:
            var = var + qparam_extract(model._modules[m])

        else:
            if hasattr(model._modules[m], 'init'):
                if hasattr(model._modules[m], 'sB'):
                    var = var + list(model._modules[m].parameters())[2:]
                else:
                    var = var + list(model._modules[m].parameters())[1:]

            # elif hasattr(model._modules[m], 'affine'):
            #     var = var + list(model._modules[m].parameters())
                    
                # var = var + list(model._modules[m].parameters())[1:12]

    return var

def param_extract(model):

    var = list()

    for m in model._modules:
        if len(model._modules[m]._modules) > 0:
            var = var + param_extract(model._modules[m])

        else:
            if hasattr(model._modules[m], 'init'):
                if hasattr(model._modules[m], 'sB'):
                    var = var + list(model._modules[m].parameters())[0:2]
                else:
                    var = var + list(model._modules[m].parameters())[0:1]

            # elif hasattr(model._modules[m], 'affine'):
            #     continue
            else:
                var = var + list(model._modules[m].parameters())

    return var

def run(config):

    student_model = get_model(config).to(device)
    print("The number of parameters : %d" % count_parameters(student_model))
    criterion = get_loss(config)

    q_param = qparam_extract(student_model)
    param = param_extract(student_model)

    optimizer = get_optimizer(config, param)
    q_optimizer = get_q_optimizer(config, q_param)

    if config.student_model.pretrain.pretrained:
        model_dict = student_model.state_dict()
        pretrained_dict = torch.load(config.student_model.pretrain.dir)['state_dict']

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        student_model.load_state_dict(model_dict)
        print('load the pretrained model')

    checkpoint = utils.checkpoint.get_initial_checkpoint(config, model_type)

    if checkpoint is not None:
        last_epoch, step = utils.checkpoint.q_load_checkpoint(student_model,
                                                            optimizer,
                                                            checkpoint, model_type)
    else:
        last_epoch, step = -1, -1

    print('student model from checkpoint: {} last epoch:{}'.format(
        checkpoint, last_epoch))

    scheduler = get_scheduler(config, optimizer, last_epoch)
    q_scheduler = get_q_scheduler(config, q_optimizer, last_epoch)

    # Data augmentation
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    dataloader = torchvision.datasets.CIFAR100
    # dataloader = torchvision.datasets.CIFAR10

    ############################# Valid set ##########################

    # trainset = dataloader(root='./data', train=True, download=True, transform=train_transform)
    # valset = dataloader(root='./data', train=True, download=True, transform=test_transform)

    # valid_size = 0.1
    # num_train = len(trainset)
    # indices = list(range(num_train))
    # split = int(np.floor(valid_size * num_train))

    # np.random.seed(42)
    # np.random.shuffle(indices)

    # train_idx, valid_idx = indices[split:], indices[:split]
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    # trainloader = DataLoader(trainset, batch_size=config.train.batch_size, sampler=train_sampler,
    #                          num_workers=config.data.num_workers, pin_memory=config.data.pin_memory)

    # testloader = DataLoader(valset, batch_size=config.eval.batch_size, sampler=valid_sampler,
    #                         num_workers=config.data.num_workers)

    ###################################################################

    ############################## Testset ############################
    
    trainset = dataloader(root='./data', train=True, download=True, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True,
                             num_workers=config.data.num_workers, pin_memory=config.data.pin_memory)

    testset = dataloader(root='./data', train=False, download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=config.eval.batch_size, shuffle=False,
                            num_workers=config.data.num_workers)

    ####################################################################

    dataloaders = {'train': trainloader,
                   'test': testloader}

    writer = SummaryWriter(config.train['student' + '_dir'])

    train(config, student_model, dataloaders, criterion, optimizer, q_optimizer,
          scheduler, q_scheduler, writer, last_epoch+1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args():
    parser = argparse.ArgumentParser(description='quantization network')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()

def main():
    global device
    global model_type
    model_type = 'student'
    import warnings
    warnings.filterwarnings("ignore")

    print('train %s network'%model_type)
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')

    config = utils.config.load(args.config_file)

    os.environ["CUDA_VISIBLE_DEVICES"]= str(config.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pprint.PrettyPrinter(indent=2).pprint(config)
    utils.prepare_train_directories(config, model_type)
    run(config)

    print('success!')

if __name__ == '__main__':
    main()