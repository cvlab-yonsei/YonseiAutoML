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
from models.OneShot_darts_ss import UniformRandomSupernet
from models.cell_operations import SearchSpaceNames 
from utils.optimizers import get_optim_scheduler
from utils.flop_benchmark import get_model_infos
from utils.meter import AverageMeter
from utils.evaluation_utils import obtain_accuracy
from utils.time_utils import time_string, convert_secs2time
from utils.config_utils import load_config
from utils.starts import prepare_seed, prepare_logger, save_checkpoint


def train(xloader, network, criterion, scheduler, w_optimizer, epoch_str, print_freq, logger):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  end = time.time()
  network.train()
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
    valid_arch = network.random_genotype()
    #valid_arch = network.random_genotype_wotrash_balanced()

    scheduler.update(None, 1.0 * step / len(xloader))
    base_targets = base_targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)
    
    # update the weights
    w_optimizer.zero_grad()
    logits = network(base_inputs.cuda(non_blocking=True), valid_arch) # ADDED cuda 
    base_loss = criterion(logits, base_targets)
    base_loss.backward()
    #torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
    w_optimizer.step()

    # record
    base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
    base_losses.update(base_loss.item(),  base_inputs.size(0))
    base_top1.update(base_prec1.item(), base_inputs.size(0))
    base_top5.update(base_prec5.item(), base_inputs.size(0))
    
    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % print_freq == 0 or step + 1 == len(xloader):
      Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=base_losses, top1=base_top1, top5=base_top5)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr)
  return base_losses.avg, base_top1.avg, base_top5.avg


def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads(xargs.workers)
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
  search_space = SearchSpaceNames[xargs.search_space_name]
  logger.log('search space : {:}'.format(search_space))

  search_model = UniformRandomSupernet(
      C=xargs.channel, layers=xargs.layers, 
      num_classes=class_num, search_space=search_space, 
      affine=False, track_running_stats=bool(xargs.track_running_stats)
  )
  logger.log(search_model)

  config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  test_batch_size = 512
  train_loader, _, valid_loader = get_nas_search_loaders(
      train_data,
      valid_data,
      xargs.dataset,
      "SuperNet/configs/",
      (config.batch_size, test_batch_size),
      0 #xargs.workers,
  )
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))
  logger.log('||||||| {:10s} ||||||| Train-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(train_loader), config.batch_size))

  num_params = 0.
  num_params += sum(p.numel() for p in search_model.parameters() if p.requires_grad)
  num_params /= 1e6
  logger.log(f"# of params (M) : {num_params:.3f}")

  w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.parameters(), config)
  logger.log('optimizer : {:}'.format(w_optimizer))
  logger.log('scheduler : {:}'.format(w_scheduler))
  logger.log('criterion   : {:}'.format(criterion))

  save_checkpoint({'search_model': search_model.state_dict()}, logger.model_dir / "seed-{:}-init.pth".format(xargs.rand_seed), logger)
  
  search_model, criterion = search_model.cuda(), criterion.cuda()

  start_epoch = 0
  start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup
  for epoch in range(start_epoch, total_epoch):
    w_scheduler.update(epoch, 0.0)
    need_time = 'Time Left: {:}'.format(convert_secs2time(epoch_time.val * (total_epoch-epoch), True))
    epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
    logger.log('\n[Search the {:}-th epoch] {:}, LR={:}'.format(epoch_str, need_time, min(w_scheduler.get_lr())))

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
  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
  parser.add_argument('--channel',            type=int,   help='The number of channels.')
  parser.add_argument('--layers',          type=int,   help='The number of cells in one stage.')
  parser.add_argument('--track_running_stats',type=int,   choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  parser.add_argument('--config_path',        type=str,   help='The path of the configuration.')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--print_freq',         type=int,   help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')
  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  main(args)
