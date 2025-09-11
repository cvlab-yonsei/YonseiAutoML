import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.genotypes import Structure
from utils.meter import AverageMeter
from utils.evaluation_utils import obtain_accuracy

from models.cnn import NetworkCIFAR as Network

CIFAR_CLASSES=10 # TODO

import torchvision.transforms as transforms
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return sum(v.numel() for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def main(args):
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.rand_seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.rand_seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.rand_seed)


  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(f"{args.save_dir}/logs/{args.save_name}.txt")
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)
  logging.info("args = %s", args)
  logging.info('gpu device = %d' % args.gpu)

  genotype = list(Structure.str2structure(args.arch))
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()

  logging.info("param size = %.3fMB", count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
  )

  train_transform, valid_transform = _data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
  train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
  valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
  logging.info("Train-Loader-Num={:4d}, batch size={:4d}".format(len(train_queue), args.batch_size))
  logging.info("Valid-Loader-Num={:4d}, batch size={:4d}".format(len(valid_queue), args.batch_size))

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  start_time = time.time()
  for epoch in range(args.epochs):
    logging.info('epoch %3d lr %.4e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    scheduler.step()
    logging.info('train_acc %.2f', train_acc)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %.2f', valid_acc)

    torch.save(model.state_dict(), f"{args.save_dir}/checkpoint/{args.save_name}.pth")
  end_time = time.time() - start_time
  logging.info("Elapsed Time: {:.1f} s".format(end_time))


def train(train_queue, model, criterion, optimizer):
  objs = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = input.cuda()
    target = target.cuda(non_blocking=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = obtain_accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %.4e %.2f %.2f', step, objs.avg, top1.avg, top5.avg)
  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = AverageMeter()
  top1 = AverageMeter() 
  top5 = AverageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = input.cuda()
      target = target.cuda(non_blocking=True)
  
      logits, _ = model(input)
      loss = criterion(logits, target)
  
      prec1, prec5 = obtain_accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)
  
      if step % args.report_freq == 0:
        logging.info('valid %03d %.4e %.2f %.2f', step, objs.avg, top1.avg, top5.avg)
  return top1.avg, objs.avg


if __name__ == '__main__':
  parser = argparse.ArgumentParser("Retrain")
  parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
  parser.add_argument('--batch_size', type=int, default=96, help='batch size')
  parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
  parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
  parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
  parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
  parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
  parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
  parser.add_argument('--layers', type=int, default=20, help='total number of layers')
  parser.add_argument('--arch', type=str, help='which architecture to use')
  
  parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
  parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
  
  parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
  parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
  
  parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
  
  parser.add_argument('--rand_seed', type=int, help='random seed')
  parser.add_argument('--save_dir', type=str)
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
  parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  args.save_name = f"seed-{args.rand_seed}-retrain-{args.arch}"
  main(args) 
