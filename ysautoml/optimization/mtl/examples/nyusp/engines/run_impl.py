import os
import subprocess
from pathlib import Path


def run_experiment(
    gpu_id=0,
    seed=0,
    weighting="GeMTL",
    arch="HPS",
    dataset_path="/dataset/nyuv2",
    scheduler="step",
    mode="train",
    log_dir="./logs"
):
    os.makedirs(log_dir, exist_ok=True)
    main_script = Path(__file__).resolve().parents[0] / "main.py"

    cmd = [
        "python3", str(main_script),
        "--weighting", weighting,
        "--arch", arch,
        "--dataset_path", dataset_path,
        "--gpu_id", str(gpu_id),
        "--seed", str(seed),
        "--scheduler", scheduler,
        "--mode", mode,
    ]

    log_file = Path(log_dir) / f"{arch}_{weighting}_seed{seed}.txt"
    print(f"[RUN_IMPL] {' '.join(cmd)} >> {log_file}")
    with open(log_file, "w") as f:
        process = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[0], stdout=f, stderr=subprocess.STDOUT)

    if process.returncode != 0:
        raise RuntimeError(f"Experiment failed with code {process.returncode}")
