from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import torch


def get_last_checkpoint(checkpoint_dir):
  checkpoints = [checkpoint
                 for checkpoint in os.listdir(checkpoint_dir)
                 if checkpoint.startswith('epoch_') and
                 checkpoint.endswith('.pth')]
  if checkpoints:
    return os.path.join(checkpoint_dir, list(sorted(checkpoints))[-1])
  return None


def get_initial_checkpoint(config, model_type):
  train_dir = config.train[model_type + '_dir']
  checkpoint_dir = os.path.join(train_dir, 'checkpoint')
  return get_last_checkpoint(checkpoint_dir)

def get_checkpoint(config, name, model_type):
  train_dir = config.train[model_type + '_dir']
  checkpoint_dir = os.path.join(train_dir, 'checkpoint')
  return os.path.join(checkpoint_dir, name)

def copy_last_n_checkpoints(config, n, name, model_type):
  train_dir = config.train[model_type + '_dir']
  checkpoint_dir = os.path.join(train_dir, 'checkpoint')
  checkpoints = [checkpoint
                 for checkpoint in os.listdir(checkpoint_dir)
                 if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth')]
  checkpoints = sorted(checkpoints)
  for i, checkpoint in enumerate(checkpoints[-n:]):
    shutil.copyfile(os.path.join(checkpoint_dir, checkpoint),
                    os.path.join(checkpoint_dir, name.format(i)))

def q_load_checkpoint(model, optimizer, checkpoint, model_type, alpha=None,
                    optimizer_alpha=None, beta=None, optimizer_beta=None):
  print('load checkpoint from', checkpoint)
  checkpoint = torch.load(checkpoint)
  checkpoint_dict = checkpoint['state_dict']
  model.load_state_dict(checkpoint_dict)  # , strict=False)

  if optimizer is not None:
    optimizer.load_state_dict(checkpoint['optimizer_dict'])

  if alpha is not None and beta is not None:
    for i in range(len(alpha)):
      alpha.pop()
      beta.pop()
    alpha += checkpoint['alpha']
    beta += checkpoint['beta']
    optimizer_alpha.load_state_dict(checkpoint['optimizer_alpha_dict'])
    optimizer_beta.load_state_dict(checkpoint['optimizer_beta_dict'])

  step = checkpoint['step'] if 'step' in checkpoint else -1
  last_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else -1

  return last_epoch, step

def load_checkpoint(model, optimizer, scheduler, q_optimizer=None, q_scheduler=None, t_optimizer=None, t_scheduler=None, gpu_n=None, checkpoint=None, model_type=None, alpha=None,
                    optimizer_alpha=None, beta=None, optimizer_beta=None):
  print('load checkpoint from', checkpoint)
  loc = 'cuda:{}'.format(gpu_n)
  checkpoint = torch.load(checkpoint, map_location=loc)
  checkpoint_dict = checkpoint['state_dict']
  model.load_state_dict(checkpoint_dict)  # , strict=False)

  if optimizer is not None:
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_dict'])

  if q_optimizer is not None:
    q_optimizer.load_state_dict(checkpoint['q_optimizer_dict'])
    q_scheduler.load_state_dict(checkpoint['q_scheduler_dict'])

  if t_optimizer is not None:
    t_optimizer.load_state_dict(checkpoint['t_optimizer_dict'])
    t_scheduler.load_state_dict(checkpoint['t_scheduler_dict'])

  step = checkpoint['step'] if 'step' in checkpoint else -1
  last_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else -1

  return last_epoch, step

def save_checkpoint(config, model, optimizer, scheduler, q_optimizer, q_scheduler, t_optimizer=None, t_scheduler=None,
                    epoch=0, step=0, model_type=None, alpha=None, optimizer_alpha=None,
                    beta=None, optimizer_beta=None, weights_dict=None, name=None):
  train_dir = config.train[model_type + '_dir']
  checkpoint_dir = os.path.join(train_dir, 'checkpoint')

  if name:
    checkpoint_path = os.path.join(checkpoint_dir, '{}.pth'.format(name))
  else:
    checkpoint_path = os.path.join(checkpoint_dir,
                                   'epoch_{:04d}.pth'.format(epoch))
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()

  if alpha is not None and beta is not None:
    weights_dict = {
      'state_dict': state_dict,
      'optimizer_dict': optimizer.state_dict(),
      'epoch': epoch,
      'step': step,
      'alpha': alpha,
      'optimizer_alpha_dict':optimizer_alpha.state_dict(),
      'beta': beta,
      'optimizer_beta_dict':optimizer_beta.state_dict(),
    }

  elif q_optimizer is None:
    weights_dict = {
      'state_dict': state_dict,
      'optimizer_dict': optimizer.state_dict(),
      'scheduler_dict': scheduler.state_dict(),
      'epoch': epoch,
      'step': step,
    }

  elif t_optimizer is None:
    weights_dict = {
      'state_dict': state_dict,
      'optimizer_dict': optimizer.state_dict(),
      'scheduler_dict': scheduler.state_dict(),
      'q_optimizer_dict': q_optimizer.state_dict(),
      'q_scheduler_dict': q_scheduler.state_dict(),
      'epoch': epoch,
      'step': step,
    }

  elif weights_dict is None:
    weights_dict = {
      'state_dict': state_dict,
      'optimizer_dict': optimizer.state_dict(),
      'scheduler_dict': scheduler.state_dict(),
      'q_optimizer_dict': q_optimizer.state_dict(),
      'q_scheduler_dict': q_scheduler.state_dict(),
      't_optimizer_dict': t_optimizer.state_dict(),
      't_scheduler_dict': t_scheduler.state_dict(),
      'epoch': epoch,
      'step': step,
    }

  torch.save(weights_dict, checkpoint_path)
