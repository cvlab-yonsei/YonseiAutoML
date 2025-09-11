from .engines.supernet_train_impl import SuperNetTrainConfig, run as _run_train
import os

def train_supernet(**kwargs):
    cfg = SuperNetTrainConfig(**kwargs)

    # fill in paths like original train.py
    cfg.save_name = f"{cfg.tag}-seed-{cfg.seed}"
    cfg.log_path  = f"{cfg.save_path}/logs/{cfg.save_name}.txt"
    cfg.ckpt_path = f"{cfg.save_path}/checkpoint/{cfg.save_name}.pt"

    os.makedirs(os.path.dirname(cfg.log_path), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.ckpt_path), exist_ok=True)

    return _run_train(cfg)


# def train_supernet_decom(**kwargs):
#     """
#     Train SuperNet with decomposition (corresponds to SuperNet/scripts/train_decom.sh).
#     Example:
#         train_supernet_decom(tag="exp2", seed=1)
#     """
#     cfg = SuperNetTrainDecomConfig(**kwargs)
#     return _run_train_decom(cfg)
