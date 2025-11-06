import os
import subprocess
from pathlib import Path
import yaml  # 🔹 추가

def train_fxp(
    config="configs/resnet20_cifar100.yml",
    device="cuda:0",
    seed=42,
    save_dir="./logs/fxp_exp",
    arch_path=None,   # ✅ 추가: best_structure.txt 경로
):
    """
    Run FXP (Fixed-point / Quantization training) using the original LBT `train.py`.
    If arch_path is provided, inject it into config.student_model.params.
    """

    base_dir = Path(__file__).resolve().parent
    train_script = base_dir / "engines" / "train.py"
    config_path = base_dir / "engines" / config
    save_path = Path(save_dir).resolve()
    os.makedirs(save_path, exist_ok=True)

    # ✅ (1) config 로드 및 수정
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if arch_path is not None:
        print(f"[FXP] Injecting arch_path into config: {arch_path}")
        if "params" not in cfg["student_model"]:
            cfg["student_model"]["params"] = {}
        cfg["student_model"]["params"]["arch_path"] = arch_path
        cfg["student_model"]["name"] = "mobilenet_ste"  # 안전하게 명시

        # 수정된 config를 임시 저장
        tmp_cfg_path = save_path / f"tmp_config_{Path(config).stem}.yml"
        with open(tmp_cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)
        config_path = tmp_cfg_path

    cmd = [
        "python3",
        str(train_script),
        "--config", str(config_path),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
    env["PYTHONPATH"] = str(base_dir / "engines")

    print(f"[FXP] Running: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        cwd=str(base_dir / "engines"),
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
