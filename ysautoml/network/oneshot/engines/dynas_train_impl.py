import os, sys, time, random, pickle, logging

# 절대경로 기준으로 oneshot 폴더를 sys.path에 추가
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ONESHOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # .../oneshot
if ONESHOT_DIR not in sys.path:
    sys.path.insert(0, ONESHOT_DIR)

print("[DEBUG] Added to sys.path:", ONESHOT_DIR)

from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from scipy import stats

# xautodl imports
from xautodl.config_utils import dict2config
from xautodl.datasets import get_datasets, get_nas_search_loaders
from xautodl.models import get_cell_based_tiny_net, get_search_spaces
from xautodl.models.cell_searchs.genotypes import Structure
from xautodl.utils import obtain_accuracy

# local utils
from .utils.get_strucs import get_struc
from .utils.get_num_params import get_num_params
from .utils.LR_scheduler import AdaptiveParamSchedule


@dataclass
class DynasTrainConfig:
    log_dir: str = "./logs/tmp"
    file_name: str = "tmp"
    seed: int = 0
    epochs: int = 250
    lr: float = 0.025
    momentum: float = 0.9
    wd: float = 0.0005
    nesterov: bool = True
    method: str = "dynas"
    max_coeff: float = 4.0
    train_batch_size: int = 64
    val_batch_size: int = 256
    save_path: str = "./results"


def prepare_seed(seed):
    if seed is not None and seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


def run(cfg: DynasTrainConfig):
    prepare_seed(cfg.seed)
    writer = SummaryWriter(cfg.log_dir)

    # ✅ NAS-Bench-201 Search Space
    search_space = get_search_spaces("cell", "nas-bench-201")
    model_config = dict2config(
        {
            "name": "RANDOM",
            "C": 16,
            "N": 5,
            "max_nodes": 4,
            "num_classes": 10,
            "space": search_space,
            "affine": False,
            "track_running_stats": False,
        },
        None,
    )

    network = get_cell_based_tiny_net(model_config).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    train_data, valid_data, _, _ = get_datasets("cifar10", "./dataset", -1)
    search_loader, _, valid_loader = get_nas_search_loaders(
        train_data, valid_data, "cifar10", "configs/nas-benchmark/",
        (cfg.train_batch_size, cfg.val_batch_size), 4,
    )

    # =============== Optimizer/Scheduler Setup ===============
    if cfg.method == "baseline":
        optimizer = torch.optim.SGD(
            params=network.parameters(),
            lr=cfg.lr, momentum=cfg.momentum,
            weight_decay=cfg.wd, nesterov=cfg.nesterov
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs * len(search_loader), eta_min=0
        )
    else:
        optimizers = [
            torch.optim.SGD(
                params=network.parameters(),
                lr=cfg.lr, momentum=cfg.momentum,
                weight_decay=cfg.wd, nesterov=cfg.nesterov,
            )
            for _ in range(5)
        ]
        schedulers = [
            AdaptiveParamSchedule(opt, epochs=cfg.epochs * 391, eta_min=0)
            for opt in optimizers
        ]
        C_min = get_num_params(get_struc()[0])
        C_max = get_num_params(get_struc()[11718])
        r_max, r_min = cfg.max_coeff, 1 / cfg.max_coeff
        w = -(r_max - r_min) / (np.log(C_max) - np.log(C_min))
        tau = r_min - w * np.log(C_max)
        get_LR_exp_coeff = lambda n: w * np.log(n) + tau

    # =============== Training ===============
    struc = get_struc()
    edge2index = network.edge2index
    op_names = deepcopy(search_space)
    max_nodes = 4

    def genotype(enc):
        genotypes = []
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                node_str = f"{i}<-{j}"
                with torch.no_grad():
                    weights = enc[edge2index[node_str]]
                    op_name = op_names[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    total_iter = 0
    split_edge = random.randrange(6)

    for ep in range(cfg.epochs):
        network.train()
        for i, (inputs, labels, _, _) in enumerate(search_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            net_num = random.randrange(15625)
            network.arch_cache = genotype(struc[net_num])

            if cfg.method == "baseline":
                optimizer.zero_grad()
                _, preds = network(inputs)
                loss = criterion(preds, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 5)
                optimizer.step()
                scheduler.step()

            else:
                for j in range(5):
                    if struc[net_num][split_edge, j] == 1:
                        num_param = get_num_params(struc[net_num])
                        schedulers[j].exp_coeff = get_LR_exp_coeff(num_param)
                        schedulers[j].cur_ep = total_iter
                        schedulers[j].step()
                        optimizers[j].zero_grad()
                        _, preds = network(inputs)
                        loss = criterion(preds, labels)
                        loss.backward()
                        nn.utils.clip_grad_norm_(network.parameters(), 5)
                        optimizers[j].step()

            top1, top5 = obtain_accuracy(preds.data, labels.data, topk=(1, 5))
            writer.add_scalar("train/subnet_loss", loss.item(), total_iter)
            writer.add_scalar("train/subnet_top1", top1, total_iter)
            writer.add_scalar("train/subnet_top5", top5, total_iter)
            total_iter += 1

        print(f"[Epoch {ep}] top1={top1.item():.3f}")

    # =============== Evaluation ===============
    with open("benchmark_data/cifar10_accs.pkl", "rb") as f:
        cifar10_accs = pickle.load(f)
    with open("benchmark_data/cifar100_accs.pkl", "rb") as f:
        cifar100_accs = pickle.load(f)
    with open("benchmark_data/imagenet_accs.pkl", "rb") as f:
        imagenet_accs = pickle.load(f)
    with open("benchmark_data/num_params.pkl", "rb") as f:
        num_params = pickle.load(f)
    with open("benchmark_data/kendal_320_idx.pkl", "rb") as f:
        eval_arch_list = pickle.load(f)

    valid_accs = []
    for i in range(len(struc)):
        network.arch_cache = genotype(struc[i])
        with torch.no_grad():
            network.eval()
            correct = 0
            total = 0
            for inputs, labels in valid_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                _, preds = network(inputs)
                _, predicted = torch.max(preds.data, 1)
                total += preds.size(0)
                correct += (predicted == labels).sum().item()
            valid_acc = correct / total
            valid_accs.append(valid_acc)
            print(f"struc={i}, valid={valid_acc*100:.2f}%, real={cifar10_accs[i]}%")

    with open(f"{cfg.save_path}/{cfg.file_name}.pkl", "wb") as f:
        pickle.dump(valid_accs, f)

    tau10, _ = stats.kendalltau(np.array(valid_accs)[eval_arch_list],
                                np.array(cifar10_accs)[eval_arch_list])
    tau100, _ = stats.kendalltau(np.array(valid_accs)[eval_arch_list],
                                 np.array(cifar100_accs)[eval_arch_list])
    tau_imnet, _ = stats.kendalltau(np.array(valid_accs)[eval_arch_list],
                                    np.array(imagenet_accs)[eval_arch_list])
    writer.add_scalar("Kendall_320/cifar10", tau10, total_iter)
    writer.add_scalar("Kendall_320/cifar100", tau100, total_iter)
    writer.add_scalar("Kendall_320/imagenet", tau_imnet, total_iter)

    print(f"Kendall CIFAR10={tau10:.4f}, CIFAR100={tau100:.4f}, ImageNet={tau_imnet:.4f}")
    writer.close()
