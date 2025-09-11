##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
######################################################################################
# One-Shot Neural Architecture Search via Self-Evaluated Template Network, ICCV 2019 #
######################################################################################
import os, sys, time, random, argparse, json
from copy import deepcopy
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from datasets.get_dataset_with_transform import get_datasets, get_nas_search_loaders
#from models.OneShot import UniformRandomSupernet
from models.OneShot_decom2 import UniformRandomSupernet_decom as UniformRandomSupernet
#from models.OneShot_decom import UniformRandomSupernet_decom as UniformRandomSupernet
#from models.OneShot_decom4 import UniformRandomSupernet_decom as UniformRandomSupernet
from models.cell_operations import SearchSpaceNames
from utils.genotypes import Structure
from utils.flop_benchmark import get_model_infos
from utils.meter import AverageMeter
from utils.evaluation_utils import obtain_accuracy
from utils.config_utils import load_config
from utils.starts import prepare_seed, prepare_logger


import tqdm
max_train_iters = 50
def valid_func(train_loader, valid_loader, network, criterion, valid_arch, logger):
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  with torch.no_grad():
#    logger.log("Clear BN statistics")
#    for m in network.modules():
#      if isinstance(m, torch.nn.BatchNorm2d):
#        m.track_running_stats = True
#        m.running_mean = torch.nn.Parameter(torch.zeros(m.num_features, device="cuda"), requires_grad=False)
#        m.running_var = torch.nn.Parameter(torch.ones(m.num_features, device="cuda"), requires_grad=False)
#    logger.log("Calibrating BNs of {}".format(valid_arch))
#    train_iter = iter(train_loader)
#    network.train()
#    for step in tqdm.tqdm(range(max_train_iters)):
#      try:
#        base_inputs,_,_,_ = next(train_iter)
#      except:
#        train_iter = iter(train_loader)
#        base_inputs,_,_,_ = next(train_iter)
#  
#      output, logits = network(base_inputs.cuda(non_blocking=True), valid_arch)
#      del base_inputs, output, logits

    network.eval()
    for step, (arch_inputs, arch_targets) in enumerate(valid_loader):
      arch_targets = arch_targets.cuda(non_blocking=True)

      # prediction
      #_, logits = network(arch_inputs.cuda(non_blocking=True), valid_arch) # ADDED cuda
      logits = network(arch_inputs.cuda(non_blocking=True), valid_arch) # ADDED cuda
      arch_loss = criterion(logits, arch_targets)
      # record
      arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
      arch_losses.update(arch_loss.item(),  arch_inputs.size(0))
      arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
      arch_top5.update(arch_prec5.item(), arch_inputs.size(0))
  return arch_losses.avg, arch_top1.avg, arch_top5.avg


def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads(xargs.workers)
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
  config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  test_batch_size = 512
  train_loader, _, _ = get_nas_search_loaders(
      train_data,
      valid_data,
      xargs.dataset,
      "SuperNet/configs/",
      (config.batch_size, test_batch_size),
      0 #xargs.workers,
  )
  valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=test_batch_size, shuffle=False, pin_memory=True, num_workers=0)
  logger.log('||||||| {:10s} ||||||| Train-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(train_loader), len(valid_loader), (config.batch_size, test_batch_size)))

  search_space = SearchSpaceNames[xargs.search_space_name]
  logger.log('search space : {:}'.format(search_space))
  search_model = UniformRandomSupernet(
      C=xargs.channel, N=xargs.num_cells, max_nodes=xargs.max_nodes,
      num_classes=class_num, search_space=search_space, 
      affine=False,
      track_running_stats=bool(xargs.track_running_stats)
  )

  criterion = torch.nn.CrossEntropyLoss()  
  logger.log('criterion   : {:}'.format(criterion))

  search_model, criterion = search_model.cuda(), criterion.cuda()
  
  ckpt_path = logger.model_dir / xargs.ckpt
  logger.log("=> loading checkpoint of '{:}'".format(ckpt_path))
  checkpoint = torch.load(ckpt_path)['search_model']
  search_model.load_state_dict(checkpoint)

  archs = []
  split_op = None
  if xargs.edge_op == 0:
    split_op = 'none'

  if xargs.edge_op == 1:
    split_op = 'skip'

  if xargs.edge_op == 2:
    split_op = 'conv1'

  if xargs.edge_op == 3:
    split_op = 'conv3'

  if xargs.edge_op == 4:
    split_op = 'avgpool'

  with open('./SuperNet/search_pool/' + split_op + '.txt', 'r') as f:
    list_archs = eval(f.read())
  for arch in list_archs:
    archs.append(Structure(arch))
  LEN = len(archs)
  logger.log('lenth of archs is', LEN)

  json_name = xargs.ckpt.split("/")[0].split(".pth")[0]
  result_dir = xargs.output_dir + f'/{json_name}_{xargs.edge_op}'
  logger.log(result_dir)

  if os.path.exists(result_dir):
    with open(result_dir, 'r') as f:
      archs_dict = json.load(f)
  else:
    #os.mkdir(xargs.output_dir)
    archs_dict = {}

  start_time = time.time()
  for i in range(LEN):
    valid_a_loss , valid_a_top1 , valid_a_top5 = valid_func(train_loader, valid_loader, search_model, criterion, archs[i], logger)
    logger.log('{:5d}/{:5d} current test-geno is {:}'.format(i, LEN, archs[i]))
    logger.log('evaluate : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}% |'.format(valid_a_loss, valid_a_top1, valid_a_top5))
    archs_dict[str(list_archs[i])] = valid_a_top1
    with open(result_dir, 'w') as f:
      json.dump(archs_dict, f)
  search_time = time.time() - start_time
  logger.log("Search Time: {:.1f} s".format(search_time))
  logger.log(result_dir)
  logger.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser("One-Shot")
  parser.add_argument('--data_path',          type=str,   help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   help='The number of cells in one stage.')
  parser.add_argument('--select_num',         type=int,   help='The number of selected architectures to evaluate.')
  parser.add_argument('--track_running_stats',type=int,   choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  parser.add_argument('--config_path',        type=str,   help='The path of the configuration.')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--output_dir',         type=str,   help='Folder to save supernet info.')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')
  parser.add_argument('--edge_op',            type=int,   help='index of the operation of the edge')
  parser.add_argument('--ckpt',               type=str,   help='pre-trained SuperNet')
  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  main(args)
