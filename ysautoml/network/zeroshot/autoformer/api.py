import subprocess
import os
from pathlib import Path

def run_search_zeroshot(
    data_path="/dataset/ILSVRC2012",
    population_num=10000,
    seed=123,
    param_limits=6,
    min_param_limits=4,
    cfg="space-T.yaml",  # ← 이름만 넣어도 됨
    output_dir="./OUTPUT/search/AZ-NAS/Tiny",
    device="0",
    relative_position=True,
    change_qkv=True,
    dist_eval=True,
    gp=True,
):
    """
    Run Zero-Shot NAS (AutoFormer AZ-NAS search)
    """

    base_dir = Path(__file__).resolve().parent
    engine_dir = base_dir / "engines" / "ImageNet_AutoFormer"
    az_script = engine_dir / "search_autoformer_az.py"

    # ✅ YAML 경로 자동 탐색
    cfg_path = Path(cfg)
    if not cfg_path.is_absolute():
        # cfg가 단순 파일명이나 상대경로면, engines 내부 experiments 폴더 기준으로 해석
        cfg_path = engine_dir / "experiments" / "search_space" / cfg
    if not cfg_path.exists():
        raise FileNotFoundError(f"YAML config not found: {cfg_path}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device

    cmd = [
        "python3", str(az_script),
        "--data-path", data_path,
        "--population-num", str(population_num),
        "--seed", str(seed),
        "--param-limits", str(param_limits),
        "--min-param-limits", str(min_param_limits),
        "--cfg", str(cfg_path),
        "--output_dir", output_dir,
    ]

    if gp: cmd.append("--gp")
    if relative_position: cmd.append("--relative_position")
    if change_qkv: cmd.append("--change_qkv")
    if dist_eval: cmd.append("--dist-eval")

    process = subprocess.Popen(
        cmd, env=env, cwd=str(engine_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    for line in process.stdout:
        print(f"[AZ-NAS] {line}", end="")

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"AZ-NAS search failed with code {process.returncode}")


def run_retrain_zeroshot(
    cfg="Tiny.yaml",
    output_dir="./OUTPUT/AZ-NAS/Tiny",
    data_path="/dataset/ILSVRC2012",
    epochs=500,
    warmup_epochs=20,
    batch_size=256,
    device="0,1,2,3,4,5,6,7",  # GPU 8개 기본
    master_port=6666,
    nproc_per_node=8,
    model_type="AUTOFORMER",
    mode="retrain",
    relative_position=True,
    change_qkv=True,
    gp=True,
    dist_eval=True,
):
    """
    Run retraining of searched AutoFormer subnet (AZ-NAS result)
    Equivalent to train_searched_result_az.sh

    Example:
        run_retrain_zeroshot(
            cfg="./experiments/AZ-NAS/Tiny.yaml",
            output_dir="./OUTPUT/AZ-NAS/Tiny-bs256x8-500ep"
        )
    """

    base_dir = Path(__file__).resolve().parent
    engine_dir = base_dir / "engines" / "ImageNet_AutoFormer"
    train_script = engine_dir / "train_subnet.py"

    # ✅ cfg 경로는 절대/상대 모두 그대로 사용
    cfg_path = str(cfg)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    env["CUDA_LAUNCH_BLOCKING"] = "1"

    cmd = [
        "python3", "-m", "torch.distributed.launch",
        f"--master_port={master_port}",
        f"--nproc_per_node={nproc_per_node}",
        "--use_env", str(train_script),
        "--data-path", data_path,
        "--epochs", str(epochs),
        "--warmup-epochs", str(warmup_epochs),
        "--batch-size", str(batch_size),
        "--mode", mode,
        "--model_type", model_type,
        "--cfg", cfg_path,  # ✅ 입력 그대로 전달
        "--output_dir", output_dir,
    ]

    if gp: cmd.append("--gp")
    if relative_position: cmd.append("--relative_position")
    if change_qkv: cmd.append("--change_qkv")
    if dist_eval: cmd.append("--dist-eval")

    print(f"\n[Running Retrain: {cfg_path}]")
    print(" ".join(cmd))

    process = subprocess.Popen(
        cmd, env=env, cwd=str(engine_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    for line in process.stdout:
        print(f"[AZ-NAS-TRAIN] {line}", end="")

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"AZ-NAS retrain failed with code {process.returncode}")

    print(f"[✅ Completed Retrain] {cfg_path}")