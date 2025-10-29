import os
import subprocess
import sys
from pathlib import Path


def train_losssearch(
    epochs: int = 100,
    lr_model: float = 0.1,
    lr_loss: float = 0.0001,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    save_dir: str = "./logs/losssearch",
    device: str = "cuda:0"
):
    """
    Run Loss Function Search (LFS) — dynamically learns loss parameters 
    using a custom criterion optimizer integrated into YSAutoML.

    Parameters
    ----------
    epochs (int, default 100): Total number of training epochs.
    lr_model (float, default 0.1): Learning rate for the model optimizer.
    lr_loss (float, default 0.0001): Learning rate for the custom loss optimizer.
    momentum (float, default 0.9): Momentum factor for SGD optimizers.
    weight_decay (float, default 0.0005): Weight decay coefficient for L2 regularization.
    save_dir (str, default "./logs/losssearch"): Directory path to store training logs and checkpoints.
    device (str, default "cuda:0"): CUDA or CPU device identifier.

    Returns
    -------
    None : 
        All logs, model weights, and updated loss parameters are saved to the specified directory.
    """

    root_dir = Path(__file__).resolve().parents[0]
    engine_path = root_dir / "engines" / "main.py"

    cmd = [
        "python3", str(engine_path),
        "--epochs", str(epochs),
        "--lr_model", str(lr_model),
        "--lr_loss", str(lr_loss),
        "--momentum", str(momentum),
        "--wd", str(weight_decay),
        "--save_dir", str(save_dir),
        "--device", device
    ]

    print("[LOSSSEARCH] Running:", " ".join(cmd))
    process = subprocess.run(cmd)

    if process.returncode != 0:
        raise RuntimeError(f"[LOSSSEARCH] Training failed with code {process.returncode}")
