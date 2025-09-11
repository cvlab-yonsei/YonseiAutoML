##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
######################################################################################
# One-Shot Neural Architecture Search via Self-Evaluated Template Network, ICCV 2019 #
######################################################################################
import os, sys, time, random, argparse
from copy import deepcopy
import torch
from pathlib import Path


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from datasets.get_dataset_with_transform import get_datasets, get_nas_search_loaders
from models.OneShot_sp import UniformRandomSupernet_splitable as UniformRandomSupernet
from models.cell_operations_op_level import SearchSpaceNames, NAS_BENCH_201
from utils.optimizers import get_optim_scheduler
from utils.flop_benchmark import get_model_infos
from utils.meter import AverageMeter
from utils.evaluation_utils import obtain_accuracy
from utils.time_utils import time_string, convert_secs2time
from utils.config_utils import load_config
from utils.starts import prepare_seed, prepare_logger, save_checkpoint

from torch.utils.tensorboard import SummaryWriter
import json

def train(xloader, network, criterion, scheduler, w_optimizer, epoch_str, print_freq, logger):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  end = time.time()
  network.train()
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
    valid_arch = network.random_genotype()

    scheduler.update(None, 1.0 * step / len(xloader))
    base_targets = base_targets.cuda(non_blocking=True)
    data_time.update(time.time() - end)

    w_optimizer.zero_grad()
    logit_list = network(base_inputs.cuda(non_blocking=True), valid_arch) # ADDED cuda 
    loss_list = []
    for logits in logit_list:
      loss_list += [criterion(logits, base_targets)]
    base_loss = min(loss_list)
    base_logit = logit_list[loss_list.index(base_loss)]
    base_loss.backward()
    w_optimizer.step()
    del logit_list, loss_list
    torch.cuda.empty_cache()

    base_prec1, base_prec5 = obtain_accuracy(base_logit.data, base_targets.data, topk=(1, 5))
    base_losses.update(base_loss.item(),  base_inputs.size(0))
    base_top1.update(base_prec1.item(), base_inputs.size(0))
    base_top5.update(base_prec5.item(), base_inputs.size(0))
    
    batch_time.update(time.time() - end)
    end = time.time()

    if step % print_freq == 0 or step + 1 == len(xloader):
      Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=base_losses, top1=base_top1, top5=base_top5)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr)
  return base_losses.avg, base_top1.avg, base_top5.avg


def get_split_info(xargs, logger):
  dir_pret = f'./SuperNet/checkpoint/seed-{xargs.rand_seed}-opt5-wotrash'
  ckpt1 = torch.load(f"{dir_pret}/seed-{xargs.rand_seed}-init.pth", map_location='cpu')['search_model']
  ckpt2 = torch.load(f"{dir_pret}/seed-{xargs.rand_seed}-last.pth", map_location='cpu')['search_model']  

  cs_dict = {}
  for k1,v1 in ckpt1.items():
    cs = torch.nn.functional.cosine_similarity(v1.flatten(), ckpt2[k1].flatten(), dim=0).item()
    cs_dict[k1] = cs

  del cs_dict['classifier.weight']
  del cs_dict['classifier.bias']
  del cs_dict['stem.0.weight']

  del cs_dict['cells.5.conv_a.op.1.weight']
  del cs_dict['cells.5.conv_b.op.1.weight']
  del cs_dict['cells.5.downsample.1.weight']

  del cs_dict['cells.11.conv_a.op.1.weight']
  del cs_dict['cells.11.conv_b.op.1.weight']
  del cs_dict['cells.11.downsample.1.weight']

  for k in list(cs_dict.keys()):
    if '2.op.1.weight' in k: del cs_dict[k]

  TOP_K = [6, 0, 0]
  cnt_shallow, cnt_middle, cnt_deep = 0, 0, 0
  split_info = {}
  for k,v in sorted(cs_dict.items(), key=lambda d: d[1], reverse=True):
    name = k.split('.')
    cell_ind = int(name[1])
    if cell_ind < 5:
      if cnt_shallow < TOP_K[0]:
        cnt_shallow += 1
        if cell_ind not in split_info:
          split_info[cell_ind] = {}
        if name[3] in split_info[cell_ind]:
          split_info[cell_ind][name[3]] += [NAS_BENCH_201[int(name[4])]]
        else:
          split_info[cell_ind][name[3]] = [NAS_BENCH_201[int(name[4])]]
      else:
        continue
    elif cell_ind > 5 and cell_ind < 11:
      if cnt_middle < TOP_K[1]:
        cnt_middle += 1
        if cell_ind not in split_info:
          split_info[cell_ind] = {}
        if name[3] in split_info[cell_ind]:
          split_info[cell_ind][name[3]] += [NAS_BENCH_201[int(name[4])]]
        else:
          split_info[cell_ind][name[3]] = [NAS_BENCH_201[int(name[4])]]
      else:
        continue
    else:
      if cnt_deep < TOP_K[2]:
        cnt_deep += 1
        if cell_ind not in split_info:
          split_info[cell_ind] = {}
        if name[3] in split_info[cell_ind]:
          split_info[cell_ind][name[3]] += [NAS_BENCH_201[int(name[4])]]
        else:
          split_info[cell_ind][name[3]] = [NAS_BENCH_201[int(name[4])]]
      else:
        continue


#  TOP_K = 11 
#  split_info = {}
#  #for k,v in sorted(cs_dict.items(), key=lambda d: d[1], reverse=False): # Bottom
#  for k,v in sorted(cs_dict.items(), key=lambda d: d[1], reverse=True): # Top
#    name = k.split('.')
#    cell_ind = int(name[1])
#    cnt = 0
#    for kkk in split_info.keys():
#      cnt += len(split_info[kkk])
#    if cell_ind > 11 and cnt < TOP_K:
#        if cell_ind not in split_info:
#          split_info[cell_ind] = {}
#        if name[3] in split_info[cell_ind]:
#          split_info[cell_ind][name[3]] += [NAS_BENCH_201[int(name[4])]]
#        else:
#          split_info[cell_ind][name[3]] = [NAS_BENCH_201[int(name[4])]]
#    else:
#      continue

#  split_info = {}
#  for k,v in cs_dict.items():
#    name = k.split('.')
#    cell_ind = int(name[1])
#    if cell_ind > 14:
#      if cell_ind not in split_info:
#        split_info[cell_ind] = {}
#      if name[3] in split_info[cell_ind]:
#        split_info[cell_ind][name[3]] += [NAS_BENCH_201[int(name[4])]]
#      else:
#        split_info[cell_ind][name[3]] = [NAS_BENCH_201[int(name[4])]]

#  TOP_K = 6 #int(xargs.top_k*len(cs_dict)) - 3 
#  split_info = {}
#  #for k,v in sorted(cs_dict.items(), key=lambda d: d[1], reverse=False)[:TOP_K]: # For Bottom-x%
#  for k,v in sorted(cs_dict.items(), key=lambda d: d[1], reverse=True)[:TOP_K]: # For Top-x%
#    name = k.split('.')
#    cell_ind = int(name[1])
#    if cell_ind not in split_info:
#      split_info[cell_ind] = {}
#    if 'edges' in k:
#      if name[3] in split_info[cell_ind]:
#        split_info[cell_ind][name[3]] += [NAS_BENCH_201[int(name[4])]]
#      else:
#        split_info[cell_ind][name[3]] = [NAS_BENCH_201[int(name[4])]]
#    else:
#      split_info[cell_ind][name[2]] = 'conv'

  for k, v in split_info.items():
    split_info[k] = dict(sorted(v.items(), key=lambda x: x[0]))
  split_info = dict(sorted(split_info.items(), key=lambda item: item[0]))
  del ckpt1, ckpt2, cs_dict

#  logger.log('split_info (# of split operations={:}) : {:}'.format(TOP_K, split_info))
  return split_info
 

def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  assert xargs.option in [1,2,3,4,5], 'Wrong option'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads(xargs.workers)
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
  config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  test_batch_size = 512
  train_loader, _, valid_loader = get_nas_search_loaders(
      train_data,
      valid_data,
      xargs.dataset,
      "SuperNet/configs/",
      (config.batch_size, test_batch_size),
      0,
  )
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))
  logger.log('||||||| {:10s} ||||||| Train-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(train_loader), len(valid_loader), config.batch_size))

  search_space = SearchSpaceNames[xargs.search_space_name]
  logger.log('search space : {:}'.format(search_space))

  split_info = get_split_info(xargs, logger)

  search_model = UniformRandomSupernet(
      C=xargs.channel, N=xargs.num_cells, max_nodes=xargs.max_nodes,
      num_classes=class_num, search_space=search_space, 
      affine=False, 
      track_running_stats=bool(xargs.track_running_stats),
      split_info=split_info,
      option=xargs.option
  )
  logger.log(search_model)

  if xargs.option in [2,3,4,5]:
    ckpt_path = logger.model_dir / f'seed-{xargs.rand_seed}-opt5-wotrash/seed-{xargs.rand_seed}-last.pth'
    logger.log("=> loading checkpoint of '{:}'".format(ckpt_path))
    checkpoint = torch.load(ckpt_path, map_location='cpu')['search_model']
    missing, unexpected = search_model.load_state_dict(checkpoint, strict=False)
    logger.log("Missing    : {:}".format(missing))
    logger.log("Unexpected : {:}".format(unexpected))
    logger.log(f"Execute Option-{xargs.option}")
    if xargs.option == 2 or xargs.option == 3:
      for cell_ind, split_dict in split_info.items():
        for edge_str, op_list in split_dict.items():
          for op_name in op_list:
            op_ind = NAS_BENCH_201.index(op_name)
            search_model.cells[cell_ind].edges[edge_str][op_ind].op[1].reset_parameters()

    if xargs.option == 4 or xargs.option == 5:
      for cell_ind, split_dict in split_info.items():
        for edge_str, op_list in split_dict.items():
          for op_name in op_list:
            op_ind = NAS_BENCH_201.index(op_name)
            pret_w = search_model.cells[cell_ind].edges[edge_str][op_ind].op[1].weight.data 
            noise = torch.randn_like(pret_w) 
            search_model.cells[cell_ind].edges[edge_str][-1].op[1].weight.data = torch.clone(pret_w) + 0.01 * noise
            pret_w += 0.01 * noise

    if xargs.option == 3 or xargs.option == 5:
      logger.log("Freeze in part")
      search_model.freeze_part()

  #w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.parameters(), config)
  ## NOTE: This is for option2b
  param_groups = search_model.get_param_groups()
  logger.log(f"# of 1st Group {len(param_groups[0])} (split ops)")
  logger.log(f"# of 2nd Group {len(param_groups[1])} (remaining ops)")
  #params = [{'params': param_groups[0]}, {'params': param_groups[1], 'lr': 1e-3}]
  #w_optimizer, w_scheduler, criterion = get_optim_scheduler(params, config)
  w_optimizer, w_scheduler, criterion = get_optim_scheduler(param_groups[0], config)
  logger.log('optimizer : {:}'.format(w_optimizer))
  logger.log('scheduler : {:}'.format(w_scheduler))
  logger.log('criterion : {:}'.format(criterion))

  save_checkpoint({'search_model': search_model.state_dict()}, logger.model_dir / "seed-{:}-init.pth".format(xargs.rand_seed), logger)
  
  search_model, criterion = search_model.cuda(), criterion.cuda()
  
  start_epoch = 0

  start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup

  for epoch in range(start_epoch, total_epoch):
    w_scheduler.update(epoch, 0.0)
    need_time = 'Time Left: {:}'.format(convert_secs2time(epoch_time.val * (total_epoch-epoch), True))
    epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
    logger.log('\n[Search the {:}-th epoch] {:}, LR={:}'.format(epoch_str, need_time, w_scheduler.get_lr()))

    search_w_loss, search_w_top1, search_w_top5 = train(train_loader, search_model, criterion, w_scheduler, w_optimizer, epoch_str, xargs.print_freq, logger)
    search_time.update(time.time() - start_time)
    logger.log('[{:}] search [base] : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s'.format(epoch_str, search_w_loss, search_w_top1, search_w_top5, search_time.sum))

    logger.log('<<<--->>> The {:}-th epoch'.format(epoch_str))

    save_checkpoint({'search_model': search_model.state_dict(),}, logger.model_dir / f"seed-{xargs.rand_seed}-ep{epoch:03d}.pth", logger)
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  save_checkpoint({'epoch': epoch + 1, 'args': deepcopy(xargs), 'search_model': search_model.state_dict(),}, logger.model_dir / f"seed-{xargs.rand_seed}-last.pth", logger)
  logger.log('\n' + '-' * 100)
  logger.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser("One-Shot")
  parser.add_argument('--data_path',          type=str,   help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')

  parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   help='The number of cells in one stage.')
  parser.add_argument('--track_running_stats',type=int,   choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  parser.add_argument('--config_path',        type=str,   help='The path of the configuration.')

  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset (tiny-nas-benchmark).')
  parser.add_argument('--print_freq',         type=int,   help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')

  parser.add_argument('--top_k', type=float, default=0.05)
  parser.add_argument('--option', type=int, default=1)
  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  main(args)
