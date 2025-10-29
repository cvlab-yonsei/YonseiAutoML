from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as edict

import os


def _get_default_config():
  c = edict()

  # dataset
  c.data = edict()
  c.data.name = 'DefaultDataset'
  c.data.dir = './data'
  c.data.params = edict()

  # student model
  c.student_model = edict()
  c.student_model.params = edict()
  c.student_model.pretrain = edict()

  # student full model
  c.student_f_model = edict()
  c.student_f_model.params = edict()
  c.student_f_model.pretrain = edict()

  # teacher model
  c.teacher_model = edict()
  c.teacher_model.params = edict()

  # quantization
  c.quantization = edict()
  c.quantization.temp = edict()
  c.quantization.alpha_init = edict()
  c.quantization.beta_init = edict()
  c.quantization.qw_values = edict()

  # train
  c.train = edict()
  c.train.dir = './result/out'
  c.train.batch_size = 64
  c.train.num_epochs = 2000
  c.train.num_grad_acc = None

  # evaluation
  c.eval = edict()
  c.eval.batch_size = 64

  # optimizer
  c.optimizer = edict()
  c.optimizer.name = 'adam'
  c.optimizer.params = edict()
  c.optimizer.gradient_clip = edict()

  # optimizer_2
  c.optimizer_2 = edict()
  c.optimizer_2.name = 'adam'
  c.optimizer_2.params = edict()
  c.optimizer_2.gradient_clip = edict()

  # optimizer_3
  c.optimizer_3 = edict()
  c.optimizer_3.name = 'adam'
  c.optimizer_3.params = edict()
  c.optimizer_3.gradient_clip = edict()

  # optimizer_4
  c.optimizer_4 = edict()
  c.optimizer_4.name = 'adam'
  c.optimizer_4.params = edict()
  c.optimizer_4.gradient_clip = edict()

  # q_optimizer
  c.q_optimizer = edict()
  c.q_optimizer.name = 'adam'
  c.q_optimizer.params = edict()
  c.q_optimizer.gradient_clip = edict()

  # b_optimizer
  c.b_optimizer = edict()
  c.b_optimizer.name = 'adam'
  c.b_optimizer.params = edict()
  c.b_optimizer.gradient_clip = edict()

  # scheduler
  c.scheduler = edict()
  c.scheduler.name = 'none'
  c.scheduler.params = edict()

  # q_scheduler
  c.q_scheduler = edict()
  c.q_scheduler.name = 'none'
  c.q_scheduler.params = edict()

  # transforms
  c.transform = edict()
  c.transform.name = 'default_transform'
  c.transform.num_preprocessor = 4
  c.transform.params = edict()

  # losses
  c.loss = edict()
  c.loss.name = None
  c.loss.params = edict()

  return c


def _merge_config(src, dst):
  if not isinstance(src, edict):
    return

  for k, v in src.items():
    if isinstance(v, edict):
      _merge_config(src[k], dst[k])
    else:
      dst[k] = v


def load(config_path):
  with open(config_path, 'r') as fid:
    # yaml_config = edict(yaml.load(fid))
    yaml_config = edict(yaml.load(fid, Loader=yaml.FullLoader)) # PyTAML version issue

  config = _get_default_config()
  _merge_config(yaml_config, config)

  set_model_weight_dirs(config)

  return config

def set_model_weight_dirs(config):

  if 'name' in config.teacher_model:
    teacher_dir = os.path.join(config.train.dir, config.teacher_model.name)
    config.train.teacher_dir = teacher_dir + config.train.teacher_dir
  if 'name' in config.student_model:
    student_dir = os.path.join(config.train.dir, config.student_model.name)
    config.train.student_dir = student_dir + config.train.student_dir

