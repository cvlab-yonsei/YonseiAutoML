from pathlib import Path
import subprocess
import os

def train_dynas(
    log_dir="logs/spos_dynamic",
    file_name="spos_dynamic",
    seed=0,
    epochs=250,
    lr=0.025,
    momentum=0.9,
    wd=0.0005,
    nesterov=True,
    train_batch_size=64,
    val_batch_size=256,
    method="dynas",       # baseline | dynas
    max_coeff=4.0,
):
    """
    Run DYNAS or SPOS baseline training from the official train_spos.py script.
    This wraps the existing DYNAS codebase and runs it via subprocess.
    """

    # --- 경로 계산 ---
    base_dir = Path(__file__).resolve().parent / "engines"   # ✅ engines 폴더로 이동
    script_path = base_dir / "train_spos.py"                 # ✅ 실제 파일 경로
    root_dir = base_dir.parent.parent.parent                 # .../YonseiAutoML

    if not script_path.exists():
        raise FileNotFoundError(f"train_spos.py not found at {script_path}")

    # --- 실행 환경 ---
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root_dir}:{env.get('PYTHONPATH', '')}"

    # --- 실행 명령어 구성 ---
    cmd = [
        "python3",
        str(script_path),
        "--log_dir", log_dir,
        "--file_name", file_name,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--momentum", str(momentum),
        "--wd", str(wd),
        "--nesterov", str(nesterov).lower(),
        "--train_batch_size", str(train_batch_size),
        "--val_batch_size", str(val_batch_size),
        "--method", method,
        "--max_coeff", str(max_coeff),
    ]

    print(f"[DYNAS] Running: {' '.join(cmd)}")

    # --- 실행 ---
    process = subprocess.Popen(
        cmd,
        cwd=str(base_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(f"[DYNAS] {line}", end="")

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"DYNAS training failed with code {process.returncode}")

    print(f"\n✅ Completed DYNAS Training — Logs saved to: {log_dir}")
