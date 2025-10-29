from pathlib import Path
import subprocess, os

def run_evol_search(
    ckpt="baseline0-seed-0",
    seed=123,
    gpu=0,
    data_path="/dataset/ILSVRC2012",
    save_path="./Search",
    search_space="proxyless",
    workers=4,
    run_calib=False,
):
    """
    Wrapper for running evolutionary search for MobileNet supernet.
    """
    # ✅ 호출 위치 기준으로 save_path를 절대경로 변환
    user_cwd = Path.cwd()
    save_dir = (user_cwd / save_path).resolve()

    # evol_search.py 기준 path
    engine_dir = Path(__file__).resolve().parent.parent  # mobilenet/
    search_script = engine_dir / "Search" / "evol_search.py"

    # ensure directories
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        "python3",
        str(search_script),
        "--ckpt", ckpt,
        "--seed", str(seed),
        "--gpu", str(gpu),
        "--data_path", data_path,
        "--save_path", str(save_dir),  # ✅ 여기 절대경로 전달
        "--search_space", search_space,
        "--workers", str(workers),
    ]
    if run_calib:
        cmd.append("--run_calib")

    env = os.environ.copy()
    process = subprocess.Popen(
        cmd,
        cwd=str(engine_dir / "Search"),  # 실행은 Search 폴더 기준으로 하되
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(f"[EVOL_SEARCH] {line}", end="")

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Evolution search failed with code {process.returncode}")

    print(f"\n✅ Completed: logs/checkpoints saved to {save_dir}")
