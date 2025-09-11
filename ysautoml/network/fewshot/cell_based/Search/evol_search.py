##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
######################################################################################
# One-Shot Neural Architecture Search via Self-Evaluated Template Network, ICCV 2019 #
######################################################################################
import os, sys, tqdm, time, random, argparse
import torch
import numpy as np

from nas_201_api  import NASBench201API as API
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from datasets.get_dataset_with_transform import get_datasets, get_nas_search_loaders
#from models.OneShot import UniformRandomSupernet
#from models.OneShot_decom import UniformRandomSupernet_decom as UniformRandomSupernet
from models.OneShot_darts_ss import UniformRandomSupernet
from models.cell_operations import SearchSpaceNames
from utils.genotypes import Structure
from utils.evaluation_utils import obtain_accuracy
from utils.config_utils import load_config
from utils.starts import prepare_seed, prepare_logger
#from utils.flop_benchmark import get_model_infos

choice = lambda x: x[np.random.randint(len(x))] if isinstance(x, tuple) else choice(tuple(x))

def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func


@no_grad_wrapper
def get_cand_err(model, cand, train_lodaer, valid_loader, logger, run_calib, max_train_iters, max_test_iters):
    if run_calib:
        logger.log('clear bn statics....')
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.track_running_stats = True
                m.running_mean = torch.nn.Parameter(torch.zeros(m.num_features, device="cuda"), requires_grad=False)
                m.running_var = torch.nn.Parameter(torch.ones(m.num_features, device="cuda"), requires_grad=False)

        logger.log('train bn with training set (BN sanitize) ....')
        model.train()
        train_iter = iter(train_loader)
        for step in tqdm.tqdm(range(max_train_iters)):
            try:
                base_inputs,_,_,_ = next(train_iter)
            except:
                train_iter = iter(train_loader)
                base_inputs,_,_,_ = next(train_iter)

            #_, logits = model(base_inputs.cuda(non_blocking=True), cand)
            logits = model(base_inputs.cuda(non_blocking=True), cand)
            del base_inputs, logits

    
    logger.log('starting test....')
    model.eval()
    top1 = 0
    top5 = 0
    total = 0
    valid_iter = iter(valid_loader)
    for step in tqdm.tqdm(range(max_test_iters)):
        try:
            inputs, targets = next(valid_iter)
        except:
            valid_iter = iter(valid_loader)
            inputs, targets = next(valid_iter)

        # _, logits = model(inputs.cuda(non_blocking=True), cand)
        logits = model(inputs.cuda(non_blocking=True), cand)

        prec1, prec5 = obtain_accuracy(logits, targets.cuda(non_blocking=True), topk=(1, 5))

        batchsize = inputs.shape[0]
        top1 += prec1.item() * batchsize
        top5 += prec5.item() * batchsize
        total += batchsize
        del inputs, targets, logits, prec1, prec5

    top1, top5 = top1 / total, top5 / total
    logger.log('top1: {:.2f} top5: {:.2f}'.format(top1, top5))
    # top1, top5 = 1 - top1 / 100, 1 - top5 / 100 ## error 로 바꾸는 것
    # logger.log('top1: {:.2f} top5: {:.2f}'.format(top1 * 100, top5 * 100))
    return top1, top5


# def get_flops(model, input_shape=(3, 224, 224)):
#     list_conv = []

#     def conv_hook(self, input, output):
#         batch_size, input_channels, input_height, input_width = input[0].size()
#         output_channels, output_height, output_width = output[0].size()

#         assert self.in_channels % self.groups == 0

#         kernel_ops = self.kernel_size[0] * self.kernel_size[
#             1] * (self.in_channels // self.groups)
#         params = output_channels * kernel_ops
#         flops = batch_size * params * output_height * output_width
#         list_conv.append(flops)

#     list_linear = []

#     def linear_hook(self, input, output):
#         batch_size = input[0].size(0) if input[0].dim() == 2 else 1
#         weight_ops = self.weight.nelement()
#         flops = batch_size * weight_ops
#         list_linear.append(flops)

#     def foo(net):
#         childrens = list(net.children())
#         if not childrens:
#             if isinstance(net, torch.nn.Conv2d):
#                 net.register_forward_hook(conv_hook)
#             if isinstance(net, torch.nn.Linear):
#                 net.register_forward_hook(linear_hook)
#             return
#         for c in childrens:
#             foo(c)

#     foo(model)
#     input = torch.autograd.Variable(
#         torch.rand(*input_shape).unsqueeze(0), requires_grad=True)
#     out = model(input)
#     total_flops = sum(sum(i) for i in [list_conv, list_linear])
#     return total_flops


class EvolutionSearcher(object):
    def __init__(self, args, model, search_space, train_loader, valid_loader, logger):
        self.population_num = args.population_num
        self.max_epochs = args.max_epochs 

        self.select_num = args.select_num # NOTE: top-k
        self.crossover_num = args.crossover_num 
        self.mutation_num  = args.mutation_num
        self.m_prob = args.m_prob

        self.flops_limit = args.flops_limit * 1e6
        self.run_calib = args.run_calib
        self.max_train_iters = args.max_train_iters
        self.max_test_iters  = args.max_test_iters

        name = args.ckpt.split("/")[0]
        self.checkpoint_name = logger.model_dir / f"{name}-runseed-{args.rand_seed}.pth"

        self.model = model
        self.op_names = search_space
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.logger = logger
        self.num_incoming_edges = 1 if len(search_space)==5 else 2

        self.vis_dict = {}
        self.candidates = []
  
        self.memory = []
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0

    def save_checkpoint(self):
        info = {}
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        torch.save(info, self.checkpoint_name)
        self.logger.log('save checkpoint to', self.checkpoint_name)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        info = torch.load(self.checkpoint_name)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']
        self.logger.log('load checkpoint from', self.checkpoint_name)
        return True



    

    

    

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        self.logger.log('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        ## each layer 에서 확률에 의해 ops 바꿈 
        # def random_func():
        #     cand = list(choice(self.keep_top_k[k]))
        #     for i in range(self.nr_layer):
        #         if np.random.random_sample() < m_prob:
        #             cand[i] = np.random.randint(self.nr_state)
        #     return tuple(cand)

        ## cell 의 each edge 에서 확률에 의해 ops 바꿈
        def random_func():
            cand_str = choice(self.keep_top_k[k])
            cand = list(Structure.str2structure(cand_str))

            genotypes = []
            for i in range(len(cand)):
                xlist = []
                for j in range(i+self.num_incoming_edges):
                    if np.random.random_sample() < self.m_prob:
                        op_name = random.choice(self.op_names)
                        xlist.append((op_name, j))
                    else:
                        xlist.append(cand[i][j])
                genotypes.append(tuple(xlist))
            arch = Structure(genotypes)
            return arch

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if self.is_legal(cand):
                cand_str = cand.tostr()
                res.append(cand_str)
                self.logger.log('mutation {}/{}: {}'.format(len(res), mutation_num, cand_str))
            else:
                continue
            
        self.logger.log('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        self.logger.log('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        ## each layer 에서 둘 중 하나로 ops 선택
        # def random_func():
        #     p1 = choice(self.keep_top_k[k])
        #     p2 = choice(self.keep_top_k[k])
        #     return tuple(choice([i, j]) for i, j in zip(p1, p2))

        def random_func():
            cand_str1 = choice(self.keep_top_k[k])
            cand_str2 = choice(self.keep_top_k[k])
            p1 = list(Structure.str2structure(cand_str1))
            p2 = list(Structure.str2structure(cand_str2))

            genotypes = []
            for i in range(len(p1)):
                xlist = []
                for j in range(i+self.num_incoming_edges):
                    xlist.append(choice([p1[i][j], p2[i][j]]))
                genotypes.append(tuple(xlist))
            arch = Structure(genotypes)
            return arch

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if self.is_legal(cand):
                cand_str = cand.tostr()
                res.append(cand_str)
                self.logger.log('crossover {}/{}: {}'.format(len(res), crossover_num, cand_str))
            else:
                continue
            
        self.logger.log('crossover_num = {}'.format(len(res)))
        return res



    def search(self):
        self.logger.log('population_num = {} select_num = {} mutation_num (w/ prob {}) = {} crossover_num = {} random_num = {} max_epochs = {}'.format(self.population_num, self.select_num, self.m_prob, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        self.get_random(self.population_num)
        while self.epoch < self.max_epochs:
            self.logger.log('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['err'])
            self.update_top_k(self.candidates, k=50,              key=lambda x: self.vis_dict[x]['err']) # NOTE: WHY ???

            # self.logger.log('epoch = {} : top {} result'.format(self.epoch, len(self.keep_top_k[50])))
            # for i, cand in enumerate(self.keep_top_k[50]):
            #     self.logger.log('No.{} {} Top-1 err = {}'.format(i + 1, cand, self.vis_dict[cand]['err']))
            #     ops = [i for i in cand]
            #     self.logger.log(ops)

            mutation  = self.get_mutation(self.select_num, self.mutation_num, self.m_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover
            self.get_random(self.population_num)
            self.epoch += 1

        self.save_checkpoint()

    def get_random(self, num):
        self.logger.log('random select ........')
        cand_iter = self.stack_random_cand( self.model.random_genotype )
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if self.is_legal(cand):
                cand_str = cand.tostr()
                self.candidates.append(cand_str)
                self.logger.log('random {}/{}: {}'.format(len(self.candidates), num, cand_str))
            else:
                continue
        self.logger.log('random_num = {}'.format(len(self.candidates)))
        
    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                cand_str = cand.tostr()
                if cand_str not in self.vis_dict:
                    self.vis_dict[cand_str] = {}
                info = self.vis_dict[cand_str] # why ??
            for cand in cands:
                yield cand

    def is_legal(self, cand):
        cand_str = cand.tostr()
        if cand_str not in self.vis_dict:
            self.vis_dict[cand_str] = {}
        
        info = self.vis_dict[cand_str]

        if 'visited' in info:
            return False

        # if 'flops' not in info:
        #     info['flops'] = get_cand_flops(cand)

        # self.logger.log(cand, info['flops'])

        # if info['flops'] > self.flops_limit:
        #     self.logger.log('flops limit exceed')
        #     return False

        info['err'] = get_cand_err(self.model, cand, self.train_loader, self.valid_loader, self.logger, self.run_calib, self.max_train_iters, self.max_test_iters)
        info['visited'] = True
        return True


    def update_top_k(self, candidates, *, k, key, reverse=True):
        # NOTE: reverse 는 acc (True) / err (False)
        assert k in self.keep_top_k
        self.logger.log('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]


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
  train_loader, _, valid_loader = get_nas_search_loaders(
      train_data,
      valid_data,
      xargs.dataset,
      "SuperNet/configs/",
      (config.batch_size, test_batch_size),
      0 
  )
  logger.log('||||||| {:10s} ||||||| Train-Loader-Num={:4d}, batch size={:4d}'.format(xargs.dataset, len(train_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Valid-Loader-Num={:4d}, batch size={:4d}'.format(xargs.dataset, len(valid_loader), test_batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

  search_space = SearchSpaceNames[xargs.search_space_name]
  logger.log('search space : {:}'.format(search_space))

#  search_model = UniformRandomSupernet(
#      C=xargs.channel, N=xargs.num_cells, max_nodes=xargs.max_nodes,
#      num_classes=class_num, search_space=search_space,
#      affine=False,
#      track_running_stats=bool(xargs.track_running_stats)
#  )
  search_model = UniformRandomSupernet(
      xargs.channel, xargs.num_cells,
      num_classes=class_num, search_space=search_space,
  )

  if xargs.arch_nas_dataset is None:
      api = None
  else:
      api = API(xargs.arch_nas_dataset)
  logger.log('create API = {:} done'.format(api))

  #search_model = torch.nn.DataParallel(search_model).cuda()
  search_model = search_model.cuda()

  ckpt_path = f"./SuperNet/checkpoint/{xargs.ckpt}"
  logger.log("=> loading checkpoint of '{:}'".format(ckpt_path))
  checkpoint = torch.load(ckpt_path)['search_model']
  search_model.load_state_dict(checkpoint)

  searcher = EvolutionSearcher(args, search_model, search_space, train_loader, valid_loader, logger)
  start_time = time.time()
  searcher.search()
  search_time = time.time() - start_time
  logger.log("Search Time: {:.1f} s".format(search_time))

  logger.log("=> Best archs")
  info = searcher.vis_dict
  cands = sorted([cand for cand in info if 'err' in info[cand]], key=lambda cand: info[cand]['err'], reverse=True)[:5] # TODO
  for ind, cand in enumerate(cands):
      t1, t5 = info[cand]['err']
      logger.log(f"{ind:3d} (Top-1/-5) {t1:.2f}/{t5:.2f} with {cand}")

  if api is not None:
      info = api.query_by_arch(cands[0], "200")
      logger.log("{:}".format(info))
      asdf = info.split("\n")
      t1 = asdf[5][-7:-2]
      t2 = asdf[7][-7:-2]
      t3 = asdf[9][-7:-2]
      logger.log(f"{t1} {t2} {t3}")

  logger.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser("SETN")
  parser.add_argument('--data_path',          type=str,   help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   help='The number of cells in one stage.')
  parser.add_argument('--track_running_stats',type=int,   choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  parser.add_argument('--config_path',        type=str,   help='The path of the configuration.')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset (tiny-nas-benchmark).')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')
  parser.add_argument('--ckpt',               type=str,   help='pre-trained SuperNet')

  parser.add_argument('--population_num', type=int, default=50)
  parser.add_argument('--max_epochs', type=int, default=20)

  parser.add_argument('--select_num', type=int, default=10)
  parser.add_argument('--crossover_num', type=int, default=25)
  parser.add_argument('--mutation_num', type=int, default=25)
  parser.add_argument('--m_prob', type=float, default=0.1)

  parser.add_argument('--flops_limit', type=float, default=330)
  parser.add_argument('--run_calib', type=int, choices=[0,1])
  parser.add_argument('--max_train_iters', type=int, default=200)
  parser.add_argument('--max_test_iters', type=int, default=40)

  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  main(args)
