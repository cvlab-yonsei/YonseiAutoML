import subprocess
from pathlib import Path
import os


def train_fxp(
    config="configs/resnet20_cifar100.yml",
    device="cuda:0",
    seed=42,
    save_dir="./logs/fxp_exp",
):
    """
    Run FXP (Fixed-point / Quantization training) using the original LBT `train.py`.
    """

    # 절대경로 기준
    base_dir = Path(__file__).resolve().parent
    train_script = base_dir / "engines" / "train.py"   # ✅ 경로 수정
    config_path = base_dir / "engines" / config        # ✅ config도 engines 안에 있음
    save_path = Path(save_dir).resolve()
    os.makedirs(save_path, exist_ok=True)

    cmd = [
        "python3",
        str(train_script),
        "--config", str(config_path),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
    env["PYTHONPATH"] = str(base_dir / "engines")      # ✅ xautodl 등 import 문제 예방

    print(f"[FXP] Running: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        cwd=str(base_dir / "engines"),  # ✅ 실행 위치를 train.py 기준으로 설정
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(f"[FXP] {line}", end="")

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"FXP training failed with code {process.returncode}")

    print(f"\n✅ FXP training complete. Logs saved to {save_dir}")
