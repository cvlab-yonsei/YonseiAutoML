import os, sys, tqdm, time, random, argparse
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.datasets import get_datasets
from models.OneShot_decom import SuperNet_decom as SuperNet
from models.OneShot import SuperNet as Dummy
from models.layers import SearchSpaceNames

CHOICE = lambda x: x[np.random.randint(len(x))] if isinstance(x, tuple) else CHOICE(tuple(x))


def obtain_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func


@no_grad_wrapper
def get_cand_err(model, cand, train_loader, valid_loader, logger, run_calib, max_train_iters, max_test_iters):
    if run_calib:
        logger.info('Clear BNs....')
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.track_running_stats = True
                m.running_mean = torch.nn.Parameter(torch.zeros(m.num_features, device="cuda"), requires_grad=False)
                m.running_var  = torch.nn.Parameter(torch.ones(m.num_features, device="cuda"), requires_grad=False)

        logger.info('Calibrate BNs....')
        model.train()
        train_iter = iter(train_loader)
        for step in tqdm.tqdm(range(max_train_iters)):
            try:
                inputs,_ = next(train_iter)
            except:
                train_iter = iter(train_loader)
                inputs,_ = next(train_iter)

            logits = model(inputs.cuda(non_blocking=True), cand)
            del inputs, logits

    
    logger.info('Start test....')
    model.eval()
    top1  = 0
    top5  = 0
    total = 0
    valid_iter = iter(valid_loader)
    for step in tqdm.tqdm(range(max_test_iters)):
        try:
            inputs, targets = next(valid_iter)
        except:
            valid_iter = iter(valid_loader)
            inputs, targets = next(valid_iter)

        logits = model(inputs.cuda(non_blocking=True), cand)

        prec1, prec5 = obtain_accuracy(logits, targets.cuda(non_blocking=True), topk=(1, 5))

        batchsize = inputs.shape[0]
        top1  += prec1.item() * batchsize
        top5  += prec5.item() * batchsize
        total += batchsize
        del inputs, targets, logits, prec1, prec5

    top1, top5 = top1 / total, top5 / total
    err1, err5 = 100 - top1, 100 - top5 
    logger.info(f'top1: {top1:.2f}  top5: {top5:.2f}')
    logger.info(f'err1: {err1:.2f}  err5: {err5:.2f}')
    return top1, top5


class EvolutionSearcher(object):
    def __init__(self, args, model, train_loader, valid_loader, logger):
        self.population_num  = args.population_num
        self.max_epochs      = args.max_epochs 

        self.select_num      = args.select_num # NOTE: top-k
        self.crossover_num   = args.crossover_num 
        self.mutation_num    = args.mutation_num
        self.m_prob          = args.m_prob

        self.flops_limit     = args.flops_limit * 1e6
        self.run_calib       = args.run_calib
        self.max_train_iters = args.max_train_iters
        self.max_test_iters  = args.max_test_iters

        self.checkpoint_name = args.ckpt_path

        self.model   = model
        self.choices = deepcopy(self.model.choices) 
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.logger = logger

        self.vis_dict   = {}
        self.candidates = []

        self.keep_top_k = {self.select_num: [], 50: []}
        self.memory = []
        self.epoch = 0

        self.dummy = Dummy(SearchSpaceNames[args.search_space], affine=False, track_running_stats=False).cuda(args.gpu)


    def save_checkpoint(self):
        info = {}
        info['memory']     = self.memory
        info['candidates'] = self.candidates
        info['vis_dict']   = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch']      = self.epoch
        torch.save(info, self.checkpoint_name)
        self.logger.info(f"Save checkpoint to {self.checkpoint_name}")

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        info = torch.load(self.checkpoint_name)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']
        self.logger.info(f"Load checkpoint from {self.checkpoint_name}")
        return True



    def get_mutation(self, k, mutation_num, m_prob):
        self.logger.info('mutation ......')
        res = []
        max_iters =  10 * mutation_num

        def random_func():
            cand = list(CHOICE(self.keep_top_k[k]))
            for ind, ops in enumerate(self.choices):
                if np.random.random_sample() < m_prob:
                    cand[ind] = np.random.choice(ops)
            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if self.is_legal(cand):
                res.append(cand)
                self.logger.info('mutation {}/{}'.format(len(res), mutation_num))
            else:
                continue
        self.logger.info('mutation_num = {}\n'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        self.logger.info('crossover ......')
        res = []
        max_iters = 10 * crossover_num

        def random_func():
            p1 = CHOICE(self.keep_top_k[k])
            p2 = CHOICE(self.keep_top_k[k])
            return tuple(CHOICE([i, j]) for i, j in zip(p1, p2))

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if self.is_legal(cand):
                res.append(cand)
                self.logger.info('crossover {}/{}'.format(len(res), crossover_num))
            else:
                continue
        self.logger.info('crossover_num = {}\n'.format(len(res)))
        return res



    def search(self):
        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            self.logger.info(f'epoch = {self.epoch}')

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['err'])
            self.update_top_k(self.candidates, k=50,              key=lambda x: self.vis_dict[x]['err']) # NOTE: WHY ???
            for i, cand in enumerate(self.keep_top_k[self.select_num]):
                cand_info = self.vis_dict[cand]
                flops     = cand_info['flops'] / 1e6
                t1, t5    = cand_info['err']
                self.logger.info(f"No.{i+1:2d}: {cand} w/ {flops:.1f}M, (Top-1/-5) acc {t1:.2f}/{t5:.2f}")
            self.logger.info("\n")

            mutation  = self.get_mutation(self.select_num, self.mutation_num, self.m_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover
            self.get_random(self.population_num)
            self.epoch += 1

        self.save_checkpoint()

    def get_random(self, num):
        self.logger.info('random select ........')
        cand_iter = self.stack_random_cand( lambda: tuple(np.random.choice(ops) for ops in self.choices) )
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if self.is_legal(cand):
                self.candidates.append(cand)
                self.logger.info('random {}/{}'.format(len(self.candidates), num))
            else:
                continue
        self.logger.info('random_num = {}\n'.format(len(self.candidates)))

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand] # why ??
            for cand in cands:
                yield cand

    def is_legal(self, cand):
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}

        info = self.vis_dict[cand]

        if 'visited' in info:
            return False

        if 'flops' not in info:
            #info['flops'] = self.model.get_flops(cand)
            info['flops'] = self.dummy.get_flops(cand) # NOTE ! ! !

        if info['flops'] > self.flops_limit:
            print(f"flops limit exceed {info['flops']/1e6:.1f}M")
            return False
        else:
            self.logger.info(f"{cand}, {info['flops']/1e6:.1f}M")

        info['err'] = get_cand_err(self.model, cand, self.train_loader, self.valid_loader, self.logger, self.run_calib, self.max_train_iters, self.max_test_iters)
        info['visited'] = True
        return True

    def update_top_k(self, candidates, *, k, key, reverse=True):
        # NOTE: reverse 는 acc (True) / err (False)
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]


def main():
    args = get_args()
    args.save_name = f"ER-{args.ckpt}-runseed-{args.seed}"
    args.log_path  = f"{args.save_path}/logs/{args.save_name}.txt"
    args.ckpt_path = f"{args.save_path}/checkpoint/{args.save_name}.pt"

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    logger = logging.getLogger("Evolutionary Search")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(args.log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    for arg in vars(args):
        logger.info(f'{arg:<20}: {getattr(args, arg)}')

    trainset, validset, train_loader, valid_loader = get_datasets(args)
    logger.info("Trainset Size: {:7d}".format(len(trainset)))
    logger.info("Validset Size: {:7d}".format(len(validset)))
    logger.info("{}".format(trainset.transform))

    search_space = SearchSpaceNames[args.search_space]
    logger.info(search_space)
    model = SuperNet(args.num_K, tuple(args.thresholds), search_space, affine=False, track_running_stats=False).cuda(args.gpu)
    logger.info(model.choices)
    pret_path = f"./SuperNet/checkpoint/{args.ckpt}.pt"
    logger.info(f"==> Loading pre-trained SuperNet '{pret_path}'")
    checkpoint = torch.load(pret_path, map_location='cpu')['state_dict']
    model.load_state_dict(checkpoint)

    searcher = EvolutionSearcher(args, model, train_loader, valid_loader, logger)

    start_time = time.time()
    searcher.search()
    end_time = time.time() - start_time
    hours = int(end_time // 3600)
    mins  = int((end_time % 3600) // 60)
    logger.info(f"ELAPSED TIME: {end_time:.1f}(s) = {hours:02d}(h) {mins:02d}(m)")

    logger.info("==> Best archs")
    info  = searcher.vis_dict
    cands = sorted([cand for cand in info if 'err' in info[cand]], key=lambda cand: info[cand]['err'], reverse=True)[:5] # TODO
    for ind, cand in enumerate(cands):
        flops  = info[cand]['flops'] / 1e6
        t1, t5 = info[cand]['err']
        logger.info(f"No.{ind+1:2d}: {cand} w/ {flops:.1f}M, (Top-1/-5) acc {t1:.2f}/{t5:.2f}")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--num_K', type=int)
    parser.add_argument('--thresholds', type=int, nargs='+')

    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--data_path', type=str, default='/dataset/imagenet')
    parser.add_argument('--save_path', type=str, default='./Search')
    parser.add_argument('--search_space', type=str, default='proxyless', choices=['proxyless', 'spos', 'greedy'])
    parser.add_argument('--valid_size', type=int, default=50000, choices=[0, 50000])
    parser.add_argument("--num_gpus", type=int, default=1, help="the number of gpus") 
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use.')
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=200)

    parser.add_argument('--population_num', type=int, default=50)
    parser.add_argument('--max_epochs', type=int, default=20)

    parser.add_argument('--select_num', type=int, default=10)
    parser.add_argument('--crossover_num', type=int, default=25)
    parser.add_argument('--mutation_num', type=int, default=25)
    parser.add_argument('--m_prob', type=float, default=0.1)

    parser.add_argument('--flops_limit', type=float, default=330)
    parser.add_argument('--run_calib', default=False, action='store_true')
    parser.add_argument('--max_train_iters', type=int, default=200)
    parser.add_argument('--max_test_iters', type=int, default=40)
    return parser.parse_args()


if __name__ == '__main__':
    main()
