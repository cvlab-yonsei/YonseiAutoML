import os
import subprocess
from pathlib import Path


def train_mtl_nyusp(
    gpu_id: int = 0,
    seed: int = 0,
    weighting: str = "GeMTL",
    arch: str = "HPS",
    dataset_path: str = "/dataset/nyuv2",
    scheduler: str = "step",
    mode: str = "train",
    save_dir: str = "./logs/nyusp"
):
    """
    Run Multi-Task Learning (NYUv2) using the LibMTL framework integrated in YSAutoML.
    This replicates Multi-Task-Learning/examples/nyusp/run.sh.
    """
    
    # ✅ base_dir은 engines 폴더가 있는 위치로 지정해야 함
    base_dir = Path(__file__).resolve().parent / "engines"
    engine_main = base_dir / "main.py"

    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        "python3", str(engine_main),
        "--weighting", weighting,
        "--arch", arch,
        "--dataset_path", dataset_path,
        "--gpu_id", str(gpu_id),
        "--seed", str(seed),
        "--scheduler", scheduler,
        "--mode", mode
    ]

    print(f"[MTL-NYUSP] Running: {' '.join(cmd)}")
    # ✅ engines 디렉터리를 cwd로 지정해야 함
    process = subprocess.run(cmd, cwd=str(base_dir))

    if process.returncode != 0:
        raise RuntimeError(f"[MTL-NYUSP] Training failed with code {process.returncode}")
