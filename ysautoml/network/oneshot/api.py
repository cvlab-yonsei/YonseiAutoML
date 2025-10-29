from .engines.dynas_train_impl import DynasTrainConfig, run as _run_train
from pathlib import Path
import os

def train_dynas(**kwargs):
    """
    Run SPOS + DYNAS SuperNet training (NAS-Bench-201).

    Parameters
    ----------
    log_dir : str
        Directory to store TensorBoard logs (relative or absolute).
    file_name : str
        Experiment name, used for saved .pkl and log file names.
    seed : int, default 0
        Random seed for reproducibility.
    method : str, {'baseline', 'dynas'}
        Whether to run vanilla SPOS or DYNAS variant.
    epochs : int, default 250
        Training epochs.
    lr : float, default 0.025
        Initial learning rate.
    wd : float, default 5e-4
        Weight decay.
    nesterov : bool, default True
        Whether to use Nesterov momentum.
    max_coeff : float, default 4.0
        Maximum dynamic scaling coefficient (γ_max).
    train_batch_size : int, default 64
    val_batch_size : int, default 256
    save_path : str, default "./results"
        Output directory (relative to caller working directory).

    Returns
    -------
    None
    """
    cfg = DynasTrainConfig(**kwargs)

    # caller 기준으로 경로 보정
    user_cwd = Path.cwd()
    save_dir = (user_cwd / Path(cfg.save_path)).resolve()
    log_dir  = (user_cwd / Path(cfg.log_dir)).resolve()

    cfg.save_path = str(save_dir)
    cfg.log_dir   = str(log_dir)

    os.makedirs(cfg.save_path, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    _run_train(cfg)
