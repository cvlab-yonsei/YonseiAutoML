import random
import pickle
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

from lib.datasets import build_dataset
from lib import utils
from model.autoformer_space import Vision_TransformerSuper
import argparse
import os
import yaml
from lib.config import cfg, update_config_from_file
from lib.samplers import RASampler

from lib.training_free import *
# from lib.training_free.dataset import *
from timm.utils.model import unwrap_model
import copy
from timm.utils import accuracy

from lib.training_free.indicators.az_nas import compute_nas_score
import scipy.stats as stats
import time, logging
import distutils.dir_util

def mkfilepath(filename):
    distutils.dir_util.mkpath(os.path.dirname(filename))

def create_logging(log_filename=None, level=logging.INFO):
    if log_filename is not None:
        mkfilepath(log_filename)
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[
                logging.StreamHandler()
            ]
        )

def decode_cand_tuple(cand_tuple):
    depth = cand_tuple[0]
    return depth, list(cand_tuple[1:depth+1]), list(cand_tuple[depth + 1: 2 * depth + 1]), cand_tuple[-1]

def sample_configs(choices):

    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    config['embed_dim'] = [random.choice(choices['embed_dim'])]*depth

    config['layer_num'] = depth
    return config

# @torch.no_grad()
def set_arc(data_loader, model, device, amp=True, choices=None, mode='super', retrain_config=None):
    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        config = sample_configs(choices=choices)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)

    return

class Searcher(object):

    def __init__(self, args, device, model, model_without_ddp, choices, train_loader, val_loader, test_loader, output_dir):
        self.device = device
        self.indicator_name=args.indicator_name
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.args = args
        self.max_epochs = args.max_epochs
        self.population_num = args.population_num
        self.parameters_limits = args.param_limits
        self.min_parameters_limits = args.min_param_limits
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.memory = []
        self.vis_dict = {}
        self.top = {}
        self.epoch = 0
        self.checkpoint_path = args.resume
        self.candidates = []
        self.top_accuracies = []
        self.cand_params = []
        self.choices = choices
        self.all_archs = []
        self.all_score_lists_dict = {}

    def save_checkpoint(self):

        info = {}
        info['top_accuracies'] = self.top_accuracies
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['epoch'] = self.epoch
        checkpoint_path = os.path.join(self.output_dir, "checkpoint-{}.pth.tar".format(self.epoch))
        torch.save(info, checkpoint_path)
        print('save checkpoint to', checkpoint_path)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            return False
        info = torch.load(self.checkpoint_path)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.epoch = info['epoch']

        print('load checkpoint from', self.checkpoint_path)
        return True

    def is_legal(self, cand):
        info = {}
        sampled_config = {}
        sampled_config['layer_num'] = cand['layer_num']
        sampled_config['mlp_ratio'] = cand['mlp_ratio']
        sampled_config['num_heads'] = cand['num_heads']
        sampled_config['embed_dim'] = cand['embed_dim']
        n_parameters = self.model_without_ddp.get_sampled_params_numel(cand)
        info['params'] = n_parameters / 10. ** 6

        if info['params'] > self.parameters_limits:
            # print('parameters limit exceed')
            return False

        if info['params'] < self.min_parameters_limits:
            # print('under minimum parameters limit')
            return False

        # print("rank:", utils.get_rank(), cand, info['params'])
        set_arc(self.val_loader, self.model_without_ddp, self.device, amp=self.args.amp, mode='retrain',
                              retrain_config=sampled_config)

        ### AZ-NAS metric
        scores = compute_nas_score(model=self.model_without_ddp, device=self.device, trainloader=None, 
                                   resolution=self.args.input_size, batch_size=self.args.batch_size)

        # append scores
        if self.all_score_lists_dict == {}:
            for k in scores.keys():
                self.all_score_lists_dict[k] = []
        for k, v in scores.items():
            self.all_score_lists_dict[k].append(v)
        self.all_archs.append(cand)

        # compute aggregated scores
        agg_rank = None
        for k in self.all_score_lists_dict.keys():
            l = len(self.all_score_lists_dict[k])
            if agg_rank is None:
                agg_rank = np.log(stats.rankdata(self.all_score_lists_dict[k]) / l)
            else:
                agg_rank = agg_rank + np.log(stats.rankdata(self.all_score_lists_dict[k]) / l)

        # update best arch
        best_idx = np.argmax(agg_rank)
        self.top['cand']=self.all_archs[best_idx]

        # logging
        if len(self.all_archs) % 100 == 0:
            elapsed_time = time.time() - self.start_timer
            logging.info("loop_count={} / {}, time={:4g}h".format(len(self.all_archs), self.population_num, elapsed_time/3600))
            logging.info(self.top['cand'])
            params = self.model_without_ddp.get_sampled_params_numel(self.top['cand']) / 10. ** 6
            logging.info("with {}M params".format(params))
            log_string = ''
            best_idx = np.argmax(agg_rank)
            worst_idx = np.argmin(agg_rank)
            for key in self.all_score_lists_dict.keys():
                _max = self.all_score_lists_dict[key][best_idx]
                _min = self.all_score_lists_dict[key][worst_idx]
                log_string += f'max_{key}={_max:4g}\t min_{key}={_min:4g}\n'
            logging.info(log_string)


        info['visited'] = True
        self.vis_dict[cand['id']] = info
        return True

    def stack_random_cand(self, random_func, *, batchsize=20):
        while True:
            cands = [random_func(i) for i in range(batchsize)]
            for cand in cands:
                if cand not in self.candidates:
                    self.vis_dict = {}
            for cand in cands:
                yield cand

    def get_random_cand(self, id):

        config = {}
        dimensions = ['mlp_ratio', 'num_heads']
        depth = random.choice(self.choices['depth'])
        for dimension in dimensions:
            config[dimension] = [random.choice(self.choices[dimension]) for _ in range(depth)]
        config['embed_dim'] = [random.choice(self.choices['embed_dim'])] * depth
        config['layer_num'] = depth
        config['id'] = id
        return config

    def get_random(self, num):
        print('random select ........')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)

    def search(self):
        self.start_timer = time.time()
        self.get_random(self.population_num)
        logging.info('Searched Architecture: ', self.top['cand'])


def get_args_parser():
    parser = argparse.ArgumentParser('Training-free Transformer Search Script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)

    # search parameters
    parser.add_argument('--indicator-name', default='az_nas', type=str)
    parser.add_argument('--max-epochs', type=int, default=1)
    parser.add_argument('--population-num', type=int, default=8000)
    parser.add_argument('--param-limits', type=float, default=23)
    parser.add_argument('--min-param-limits', type=float, default=3)

    # config file
    parser.add_argument('--cfg',help='experiment configure file name',required=True,type=str)

    # custom parameters
    parser.add_argument('--platform', default='pai', type=str, choices=['itp', 'pai', 'aml'],
                        help='Name of model to train')
    parser.add_argument('--teacher_model', default='', type=str,
                        help='Name of teacher model to train')
    parser.add_argument('--relative_position', action='store_true')
    parser.add_argument('--max_relative_position', type=int, default=14, help='max distance in relative position embedding')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--change_qkv', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # custom model argument
    parser.add_argument('--rpe_type', type=str, default='bias', choices=['bias', 'direct'])
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_abs_pos', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr-power', type=float, default=1.0,
                        help='power of the polynomial lr scheduler')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='/dataset/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR10', 'CIFAR100', 'IMNET'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--output_dir', default='./OUTPUT/sample',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.set_defaults(amp=True)
    return parser

def main(args):

    update_config_from_file(args.cfg)
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    with open(os.path.join(args.output_dir, "config.yaml"), 'w') as f:
        f.write(args_text)

    create_logging(os.path.join(args.output_dir, "search.log"))
    logging.info(args)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args, folder_name="subImageNet")
    dataset_test, _ = build_dataset(is_train=False, args=args, folder_name="val")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=int(args.batch_size*2),
        sampler=sampler_test, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=int(args.batch_size),
        sampler=sampler_val, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )
    logging.info(cfg)
    model = Vision_TransformerSuper(img_size=args.input_size,
                                    patch_size=args.patch_size,
                                    embed_dim=cfg.SUPERNET.EMBED_DIM, depth=cfg.SUPERNET.DEPTH,
                                    num_heads=cfg.SUPERNET.NUM_HEADS,mlp_ratio=cfg.SUPERNET.MLP_RATIO,
                                    qkv_bias=True, drop_rate=args.drop,
                                    drop_path_rate=args.drop_path,
                                    gp=args.gp,
                                    num_classes=args.nb_classes,
                                    max_relative_position=args.max_relative_position,
                                    relative_position=args.relative_position,
                                    change_qkv=args.change_qkv, abs_pos=not args.no_abs_pos)

    model.to(device)
    model_without_ddp =copy.deepcopy(model)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        model_pe = copy.deepcopy(model_without_ddp)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('number of params: {}'.format(n_parameters))
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        print("resume from checkpoint: {}".format(args.resume))
        model_without_ddp.load_state_dict(checkpoint['model'])

    choices = {'num_heads': cfg.SEARCH_SPACE.NUM_HEADS, 'mlp_ratio': cfg.SEARCH_SPACE.MLP_RATIO,
               'embed_dim': cfg.SEARCH_SPACE.EMBED_DIM , 'depth': cfg.SEARCH_SPACE.DEPTH}


    t = time.time()
    searcher = Searcher(args, device, model, model_without_ddp, choices, data_loader_train, data_loader_val, data_loader_test, args.output_dir)

    searcher.search()

    best_arch = searcher.top['cand']
    output_dict = {'RETRAIN':{}}
    output_dict['RETRAIN']['MLP_RATIO'] = best_arch['mlp_ratio']
    output_dict['RETRAIN']['NUM_HEADS'] = best_arch['num_heads']
    output_dict['RETRAIN']['DEPTH'] = best_arch['layer_num']
    output_dict['RETRAIN']['EMBED_DIM'] = best_arch['embed_dim'][0]
    output_text = yaml.safe_dump(output_dict, default_flow_style=False)
    with open(os.path.join(args.output_dir, "best_arch.yaml"), 'w') as f:
        f.write(output_text)

    logging.info('total searching time = {:.2f} hours'.format(
        (time.time() - t) / 3600))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training-free Transformer Search', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
