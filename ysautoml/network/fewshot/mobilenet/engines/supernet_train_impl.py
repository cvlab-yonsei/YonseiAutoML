import os
import sys
import time
import logging
import random
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from torch.utils.tensorboard import SummaryWriter

# 패키지 내부 경로 기준
from ..utils import comm
from ..utils.datasets import get_datasets
from ..utils.optimizers import get_optimizer_scheduler
from ..utils.losses import get_losses
from ..utils.evaluator import Evaluator  # (필요 시)
from ..models.OneShot import SuperNet
from ..models.layers import SearchSpaceNames


# -------------------------
# Config (train.sh 인자 집합)
# -------------------------
@dataclass
class SuperNetTrainConfig:
    tag: str
    seed: int = -1
    thresholds: tuple = (38, 40)
    data_path: str = "/dataset/ILSVRC2012"
    save_path: str = "./SuperNet"
    search_space: str = "proxyless"
    valid_size: int = 50000
    num_gpus: int = 2
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



# -------------------------
# Sampler (원본 로직 그대로)
# -------------------------
class Sampling:
    def __init__(self, choices, thresholds, llimit=290, ulimit=330):
        self.choices    = choices
        self.thresholds = thresholds
        self.history    = {t: 0 for t in range(2)}
        self.llimit = llimit
        self.ulimit = ulimit

    def uni_sampling(self):
        return [np.random.choice(ops) for ops in self.choices]

    def flop_sampling(self, timeout=500):
        # get_flops 가 프로젝트 다른 곳에 있다면 연결하세요.
        def get_flops(_cand):
            raise NotImplementedError("FLOPs sampler is not wired in this build.")
        for _ in range(timeout):
            cand = self.uni_sampling()
            flop = get_flops(cand) * 1e6
            if self.llimit <= flop <= self.ulimit:
                return cand
        return self.uni_sampling()

    def tbs_sampling(self, timeout=500):
        group_ind = min(self.history, key=self.history.get)
        for _ in range(timeout):
            cand = self.uni_sampling()

            # Proxy: ENN (원본과 동일한 하드코드)
            ENN  = 2 * (21 - cand.count(6))

            if ENN == 38:
                g_ind = 0
            elif ENN == 40:
                g_ind = 1
            else:
                g_ind = 2

            if g_ind == group_ind:
                self.history[group_ind] += 1
                return cand
        return self.uni_sampling()


# -------------------------
# Core train loop
# -------------------------
def _do_train(args, model, logger):
    print(">>> [DEBUG] Entering training loop", flush=True)

    trainset, validset, train_loader, valid_loader = get_datasets(args)
    print("len(train_loader):", len(train_loader))
    print("len(trainset):", len(trainset))

    logger.info("Trainset Size: {:7d}".format(len(trainset)))
    logger.info("Validset Size: {:7d}".format(len(validset)))
    logger.info("{}".format(trainset.transform))

    iters_per_epoch = len(train_loader)
    args.max_iter   = iters_per_epoch * args.max_epoch

    optimizer, scheduler = get_optimizer_scheduler(args, model)
    criterion = get_losses(args).cuda(args.gpu)

    logger.info(f"--> START {args.save_name}")
    model.train()
    storages = {"CE": 0.0}
    interval_iter_verbose = max(1, iters_per_epoch // 10)

    cand_sampler = Sampling(model.module.choices if isinstance(model, DDP) else model.choices,
                            tuple(args.thresholds))
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

        rand_arch = cand_sampler.tbs_sampling()
        if args.num_gpus > 1:
            rand_arch = torch.tensor(rand_arch).cuda(args.gpu)
            torch.distributed.broadcast(rand_arch, 0)
            comm.synchronize()

        logits = model(img.cuda(args.gpu, non_blocking=True), rand_arch)
        loss = criterion(logits, gt.cuda(args.gpu, non_blocking=True))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        storages["CE"] += loss.item()

        if writer is not None:
            writer.add_scalar('loss', loss.item(), it)

        if it % interval_iter_verbose == 0:
            logger.info(f"iter: {it:5d}/{args.max_iter:5d}  CE: {loss.item():.4f}")

        if it % iters_per_epoch == 0:
            for k in storages.keys():
                storages[k] /= iters_per_epoch
            logger.info(f"--> epoch: {ep:3d}/{args.max_epoch:3d}  "
                        f"avg CE: {storages['CE']:.4f}  lr: {scheduler.get_last_lr()[0]}")
            for k in storages.keys():
                storages[k] = 0.0
            if args.num_gpus > 1:
                train_loader.sampler.set_epoch(ep)
            ep += 1

    if comm.is_main_process():
        if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
            ckpt = model.module.state_dict()
        else:
            ckpt = model.state_dict()
        info = {"state_dict": ckpt, "args": args}
        torch.save(info, args.ckpt_path)
    logger.info(f"--> END {args.save_name}")
    logger.info(cand_sampler.history)
    if writer is not None:
        writer.close()


def _main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    logger = logging.getLogger("SuperNet Training")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if comm.is_main_process():
        formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                                      datefmt="%m/%d %H:%M:%S")
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG); ch.setFormatter(formatter)
        logger.addHandler(ch)
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
        fh = logging.FileHandler(args.log_path)
        fh.setLevel(logging.DEBUG); fh.setFormatter(formatter)
        logger.addHandler(fh)
    for k, v in vars(args).items():
        logger.info(f'{k:<20}: {v}')

    search_space = SearchSpaceNames[args.search_space]
    logger.info(search_space)
    model = SuperNet(search_space, affine=False, track_running_stats=False).cuda(args.gpu)
    if args.num_gpus > 1:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
    logger.info(model)

    start_time = time.time()
    _do_train(args, model, logger)
    elapsed = time.time() - start_time
    hh = int(elapsed // 3600); mm = int((elapsed % 3600) // 60)
    logger.info(f"ELAPSED TIME: {elapsed:.1f}(s) = {hh:02d}(h) {mm:02d}(m)")
    args._elapsed = elapsed


def run(cfg: SuperNetTrainConfig):
    print(">>> [DEBUG] run() called with cfg:", cfg, flush=True)

    # --- Seed 설정 ---
    if cfg.seed is not None and cfg.seed >= 0:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        os.environ["PYTHONHASHSEED"] = str(cfg.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    # --- 경로/이름 설정 ---
    cfg.save_name = f"{cfg.tag}-seed-{cfg.seed}"
    cfg.log_path  = f"{cfg.save_path}/logs/{cfg.save_name}.txt"
    cfg.ckpt_path = f"{cfg.save_path}/checkpoint/{cfg.save_name}.pt"

    # --- 분산 학습 여부 ---
    ngpus_per_node = torch.cuda.device_count()
    if cfg.num_gpus > ngpus_per_node:
        raise ValueError(f"Requested {cfg.num_gpus} GPUs but only {ngpus_per_node} available.")

    cfg.world_size = cfg.num_gpus
    cfg.distributed = cfg.num_gpus > 1

    print(f">>> [DEBUG] distributed={cfg.distributed}, world_size={cfg.world_size}, num_gpus={cfg.num_gpus}", flush=True)

    # --- 분산/단일 GPU 실행 ---
    if cfg.distributed:
        mp.spawn(_main_worker, nprocs=cfg.num_gpus, args=(cfg.num_gpus, cfg))
    else:
        gpu_id = cfg.gpu if cfg.gpu is not None else 0
        _main_worker(gpu_id, cfg.num_gpus, cfg)
