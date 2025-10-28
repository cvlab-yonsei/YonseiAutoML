# ysautoml/network/fewshot/mobilenet/engines/supernet_train_decom_impl.py

import os, sys, time, logging, random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import utils.comm as comm
from utils.datasets import get_datasets
from utils.optimizers import get_optimizer_scheduler
from utils.losses import get_losses
from models.OneShot_decom_non import SuperNet_decom as SuperNet
from models.layers import SearchSpaceNames

# ----------------------
# Config dataclass
# ----------------------
@dataclass
class SuperNetDecomConfig:
    tag: str
    seed: int = 0
    num_K: int = 4
    thresholds: tuple = (36, 38, 40)

    data_path: str = "/dataset/ILSVRC2012"
    save_path: str = "./SuperNet"
    search_space: str = "proxyless"
    valid_size: int = 50000

    num_gpus: int = 1
    workers: int = 4
    interval_ep_eval: int = 8
    train_batch_size: int = 1024
    test_batch_size: int = 256
    max_epoch: int = 120
    learning_rate: float = 0.12
    momentum: float = 0.9
    weight_decay: float = 4e-5
    nesterov: bool = False
    lr_schedule_type: str = "cosine"
    warmup: bool = False
    label_smooth: float = 0.1

    rank: int = 0
    gpu: int = None
    dist_url: str = "tcp://127.0.0.1:23456"
    distributed: bool = False
    world_size: int = 1
    log_path: str = None
    ckpt_path: str = None
    save_name: str = None
    log_to_tb: bool = True


# ----------------------
# Sampling
# ----------------------
class Sampling:
    def __init__(self, choices, thresholds, llimit=290, ulimit=330):
        self.choices = choices
        self.thresholds = thresholds
        self.history = dict((t, 0) for t in range(1 + len(thresholds)))
        self.llimit = llimit
        self.ulimit = ulimit

    def uni_sampling(self):
        return [np.random.choice(ops) for ops in self.choices]

    def TBS_sampling(self, timeout=500):
        output = []
        for ind in range(1 + len(self.thresholds)):
            for _ in range(timeout):
                cand = self.uni_sampling()
                ENN = 2 * (21 - cand.count(6))  # NOTE: hard-coded
                if ind < len(self.thresholds):
                    if ENN == self.thresholds[ind]:
                        output.append(cand)
                        break
                else:
                    if not (ENN in self.thresholds):
                        output.append(cand)
                        break
        return output


# ----------------------
# Training loop
# ----------------------
def _do_train(args, model, logger):
    trainset, validset, train_loader, valid_loader = get_datasets(args)
    logger.info(f"Trainset Size: {len(trainset):7d}")
    logger.info(f"Validset Size: {len(validset):7d}")
    logger.info(f"{trainset.transform}")

    iters_per_epoch = len(train_loader)
    args.max_iter = iters_per_epoch * args.max_epoch

    optimizer, scheduler = get_optimizer_scheduler(args, model)
    criterion = get_losses(args).cuda(args.gpu)

    logger.info(f"--> START {args.save_name}")
    model.train()
    storages = {"CE": 0}
    interval_iter_verbose = iters_per_epoch // 10

    cand_sampler = Sampling(model.module.choices, tuple(args.thresholds))
    logger.info(cand_sampler.history)

    writer = None
    if args.log_to_tb and comm.is_main_process():
        writer = SummaryWriter(f'./tb_logs/supernet/{args.tag}')

    ep = 1
    train_iters = iter(train_loader)
    for it in range(1, args.max_iter + 1):
        try:
            img, gt = next(train_iters)
        except StopIteration:
            train_iters = iter(train_loader)
            img, gt = next(train_iters)

        rand_archs = cand_sampler.TBS_sampling()
        optimizer.zero_grad()
        for r_a in rand_archs:
            logits = model(img.cuda(args.gpu, non_blocking=True), r_a)
            loss = criterion(logits, gt.cuda(args.gpu, non_blocking=True))
            loss.backward()

        optimizer.step()
        scheduler.step()
        storages["CE"] += loss.item()

        if writer:
            writer.add_scalar("loss", loss.item(), it)

        if it % interval_iter_verbose == 0:
            logger.info(f"iter: {it:5d}/{args.max_iter:5d}  CE: {loss.item():.4f}")

        if it % iters_per_epoch == 0:
            for k in storages.keys():
                storages[k] /= iters_per_epoch
            logger.info(f"--> epoch: {ep:3d}/{args.max_epoch:3d}  avg CE: {storages['CE']:.4f}")
            for k in storages.keys():
                storages[k] = 0
            if args.num_gpus > 1:
                train_loader.sampler.set_epoch(ep)
            ep += 1

    if comm.is_main_process():
        ckpt = model.module.state_dict() if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)) else model.state_dict()
        info = {"state_dict": ckpt, "args": args}
        torch.save(info, args.ckpt_path)

    logger.info(f"--> END {args.save_name}")
    logger.info(cand_sampler.history)
    if writer:
        writer.close()


# ----------------------
# Worker
# ----------------------
def _main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend="nccl", init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank
        )

    logger = logging.getLogger("SuperNetDecom Training")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if comm.is_main_process():
        formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        fh = logging.FileHandler(args.log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    search_space = SearchSpaceNames[args.search_space]
    model = SuperNet(args.num_K, tuple(args.thresholds), search_space, affine=False, track_running_stats=False).cuda(args.gpu)
    logger.info(model.choices)
    if args.num_gpus > 1:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)

    start_time = time.time()
    _do_train(args, model, logger)
    end_time = time.time() - start_time
    logger.info(f"ELAPSED TIME: {end_time:.1f}(s)")


# ----------------------
# Entry
# ----------------------
def run(cfg: SuperNetDecomConfig):
    torch.cuda.empty_cache()
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        os.environ["PYTHONHASHSEED"] = str(cfg.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    cfg.save_name = f"{cfg.tag}-seed-{cfg.seed}"
    cfg.log_path = f"{cfg.save_path}/logs/{cfg.save_name}.txt"
    cfg.ckpt_path = f"{cfg.save_path}/checkpoint/{cfg.save_name}.pt"

    num_machines = 1
    ngpus_per_node = torch.cuda.device_count() if cfg.num_gpus is None else cfg.num_gpus
    cfg.world_size = num_machines * ngpus_per_node
    cfg.distributed = cfg.world_size > 1

    if cfg.distributed:
        mp.spawn(_main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    else:
        _main_worker(cfg.gpu or 0, ngpus_per_node, cfg)

    return {
        "save_name": cfg.save_name,
        "log_path": cfg.log_path,
        "ckpt_path": cfg.ckpt_path,
    }
