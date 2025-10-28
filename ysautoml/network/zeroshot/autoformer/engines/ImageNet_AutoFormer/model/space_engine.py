import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch

from timm.data import Mixup
from timm.utils import ModelEma
from lib import utils
import random
import time
from lib.flops import count_flops
from thop import profile
import logging

import torch.distributed as dist

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':4g', disp_avg=True):
        self.name = name
        self.fmt = fmt
        self.disp_avg = disp_avg
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}{val' + self.fmt + '}'
        fmtstr = fmtstr.format(name=self.name, val=self.val)
        if self.disp_avg:
            fmtstr += '({avg' + self.fmt + '})'
            fmtstr = fmtstr.format(avg=self.avg)
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        ## one hot to index
        if len(target.size()) == 2:
            target = target.argmax(dim=1)
        ##
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def sample_configs(choices):

    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    config['embed_dim'] = [random.choice(choices['embed_dim'])]*depth

    config['layer_num'] = depth
    return config

def train_one_epoch(writer, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, model_type, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None, args=None):
    losses = AverageMeter('Loss ', ':6.4g')
    top1 = AverageMeter('Acc@1 ', ':6.2f')
    top5 = AverageMeter('Acc@5 ', ':6.2f')
    
    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    if mode == 'retrain' and model_type == 'AUTOFORMER':
        config = retrain_config
        model_module = unwrap_model(model)
        logging.info(config)
        model_module.set_sample_config(config=config)
        logging.info(model_module.get_sampled_params_numel(config))
        # logging.info("FLOPS is {}".format(count_flops(model_module)))
        tmp = float(model_module.get_complexity(model_module.patch_embed_super.num_patches))
        logging.info("FLOPS is {}".format(tmp))

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # sample random config
        if mode == 'super' and model_type == 'AUTOFORMER':
            config = sample_configs(choices=choices)
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        elif mode == 'retrain' and model_type == 'AUTOFORMER':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if amp:
            with torch.cuda.amp.autocast():
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs = model(samples)
                    loss = 1/2 * criterion(outputs, targets) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    outputs = model(samples)
                    loss = criterion(outputs, targets)
        else:
            outputs = model(samples)
            if teacher_model:
                with torch.no_grad():
                    teach_output = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()
        
        ### avoid NaN
        flag_tensor = torch.zeros(1).to(device)
        if not math.isfinite(loss_value):
            # print("Loss is {}, stopping training".format(loss_value))
            # logging.info("Loss is {}, stopping training".format(loss_value))
            # sys.exit(1)
            flag_tensor += 1
            print("Warning: Loss is {}, setting amp=False".format(loss_value))
            logging.info("Warning: Loss is {}, setting amp=False".format(loss_value))
        dist.all_reduce(flag_tensor,op=dist.ReduceOp.SUM)
        if flag_tensor != 0:
            torch.nan_to_num(loss)
            loss.backward()
            optimizer.zero_grad()
            amp=False
            if args is not None:
                args.amp = False

            ### retry
            outputs = model(samples)
            if teacher_model:
                with torch.no_grad():
                    teach_output = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
            else:
                loss = criterion(outputs, targets)
            loss_value = loss.item()
        ###

        ### logging
        input_size = int(samples.size(0))
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        top1.update(float(acc1[0]), input_size)
        top5.update(float(acc5[0]), input_size)
        losses.update(loss_value, input_size)
        if (i % print_freq == 0) and (i>0) and (writer is not None):
            writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], len(data_loader)*epoch + i)
            writer.add_scalar('train/loss(current)', float(loss_value), len(data_loader)*epoch + i)
            writer.add_scalar('train/loss(average)', losses.avg, len(data_loader)*epoch + i)
            writer.add_scalar('train/top1(average)', top1.avg, len(data_loader)*epoch + i)
            writer.add_scalar('train/top5(average)', top5.avg, len(data_loader)*epoch + i)
        ###

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logging.info("Averaged stats:{}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(writer, epoch, data_loader, model_type, model, device, amp=True, choices=None, mode='super', retrain_config=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    if mode == 'super' and model_type == 'AUTOFORMER':
        config = sample_configs(choices=choices)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    elif model_type == 'AUTOFORMER':
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config

    if model_type == 'AUTOFORMER':
        logging.info("sampled model config: {}".format(config))
        parameters = model_module.get_sampled_params_numel(config)
        logging.info("sampled model parameters: {}".format(parameters))

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logging.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    if writer is not None:
        writer.add_scalar('val/top1', metric_logger.acc1.global_avg, epoch)
        writer.add_scalar('val/top5', metric_logger.acc5.global_avg, epoch)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}