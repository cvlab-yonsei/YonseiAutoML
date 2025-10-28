import subprocess
import os
from pathlib import Path


def run_search_zeroshot(
    gpu=0,
    seed=123,
    metric="AZ_NAS",
    population_size=1024,
    evolution_max_iter=int(1e5),
    resolution=224,
    budget_flops=1e9,
    max_layers=16,
    batch_size=64,
    data_path="/dataset/ILSVRC2012/",
    num_classes=1000,
):
    """
    Run zero-shot NAS search for MobileNetV2 (ImageNet_MBV2)
    Equivalent to the evolution_search_az.sh script.

    Example:
        run_search_mbv2(gpu=0, seed=123)
    """

    base_dir = Path(__file__).resolve().parent
    engine_dir = base_dir / "engines" / "ImageNet_MBV2"

    evolution_script = engine_dir / "evolution_search_az.py"
    analyze_script = engine_dir / "analyze_model.py"
    search_space = engine_dir / "SearchSpace" / "search_space_IDW_fixfc.py"

    # save directory 설정
    save_dir = engine_dir / "save_dir" / f"{metric}_flops1G-searchbs{batch_size}-pop{population_size}-iter{evolution_max_iter}-{seed}"
    os.makedirs(save_dir, exist_ok=True)

    # 초기 구조 파일 생성
    init_plainnet_path = save_dir / "init_plainnet.txt"
    init_plainnet_text = (
        "SuperConvK3BNRELU(3,8,2,1)"
        "SuperResIDWE6K3(8,32,2,8,1)"
        "SuperResIDWE6K3(32,48,2,32,1)"
        "SuperResIDWE6K3(48,96,2,48,1)"
        "SuperResIDWE6K3(96,128,2,96,1)"
        "SuperConvK1BNRELU(128,2048,1,1)"
    )
    with open(init_plainnet_path, "w") as f:
        f.write(init_plainnet_text)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Evolution Search 명령어
    search_cmd = [
        "python3", str(evolution_script),
        "--gpu", str(gpu),
        "--zero_shot_score", metric,
        "--search_space", str(search_space),
        "--budget_flops", str(budget_flops),
        "--max_layers", str(max_layers),
        "--batch_size", str(batch_size),
        "--input_image_size", str(resolution),
        "--plainnet_struct_txt", str(init_plainnet_path),
        "--num_classes", str(num_classes),
        "--evolution_max_iter", str(int(evolution_max_iter)),
        "--population_size", str(population_size),
        "--save_dir", str(save_dir),
        "--dataset", "imagenet-1k",
        "--num_worker", "0",
        "--rand_input", "True",
        "--search_no_res", "False",
        "--seed", str(seed),
        "--datapath", data_path,
    ]

    print(f"\n[Running MBV2 Search] {' '.join(search_cmd)}")

    process = subprocess.Popen(
        search_cmd, cwd=str(engine_dir),
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    for line in process.stdout:
        print(f"[MBV2-SEARCH] {line}", end="")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"MBV2 search failed with code {process.returncode}")

    # analyze_model.py 실행
    analyze_cmd = [
        "python3", str(analyze_script),
        "--input_image_size", str(resolution),
        "--num_classes", str(num_classes),
        "--arch", "Masternet.py:MasterNet",
        "--plainnet_struct_txt", str(save_dir / "best_structure.txt"),
    ]

    print(f"\n[Running MBV2 Analyze] {' '.join(analyze_cmd)}")

    process = subprocess.Popen(
        analyze_cmd, cwd=str(engine_dir),
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    for line in process.stdout:
        print(f"[MBV2-ANALYZE] {line}", end="")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"MBV2 analyze failed with code {process.returncode}")

    print(f"[✅ Completed Search & Analyze] → {save_dir}")
    return save_dir


def run_retrain_zeroshot(
    gpu_devices="0,1,2,3,4,5,6,7",
    metric="AZ_NAS",
    population_size=1024,
    evolution_max_iter=int(1e5),
    seed=123,
    num_workers=12,
    init="custom_kaiming",
    epochs=150,
    resolution=224,
    batch_size_per_gpu=64,
    world_size=8,
    data_path="/dataset/ILSVRC2012/",
    best_structure_path=None,  # 직접 경로 지정 가능 (상대/절대 둘 다)
):
    """
    Run retraining of searched MobileNetV2 (AZ-NAS result)
    Equivalent to train_AZ-NAS_flops1G-150ep-bs64x8.sh

    Example:
        # 기본 search 결과 기반 retrain
        run_retrain_zeroshot()

        # 상대경로로 best_structure 지정
        run_retrain_zeroshot(best_structure_path="save_dir/AZ_NAS_flops600M/best_structure.txt")

        # 절대경로로 지정
        run_retrain_zeroshot(best_structure_path="/data2/hyunju/exp/best_structure.txt")
    """

    base_dir = Path(__file__).resolve().parent
    engine_dir = base_dir / "engines" / "ImageNet_MBV2"
    analyze_script = engine_dir / "analyze_model.py"
    train_script = engine_dir / "train_image_classification.py"

    # 기본 save_dir 설정
    save_dir = (
        engine_dir
        / "save_dir"
        / f"{metric}_flops1G-searchbs64-pop{population_size}-iter{int(evolution_max_iter)}-{seed}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # best_structure_path가 주어졌다면 상대경로도 허용
    if best_structure_path:
        plainnet_txt = Path(best_structure_path)
        if not plainnet_txt.is_absolute():
            # 실행 중인 스크립트(api_test_network.py) 기준으로 상대경로 해석
            plainnet_txt = (Path(os.getcwd()) / best_structure_path).resolve()
    else:
        plainnet_txt = save_dir / "best_structure.txt"


    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_devices
    env["HOROVOD_FUSION_THRESHOLD"] = "67108864"

    # 1 analyze_model.py 실행
    analyze_cmd = [
        "python3",
        str(analyze_script),
        "--input_image_size",
        str(resolution),
        "--num_classes",
        "1000",
        "--arch",
        "Masternet.py:MasterNet",
        "--plainnet_struct_txt",
        str(plainnet_txt),
    ]

    print(f"\n[Running Analyze] {' '.join(analyze_cmd)}")

    analyze_proc = subprocess.Popen(
        analyze_cmd,
        cwd=str(engine_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in analyze_proc.stdout:
        print(f"[MBV2-ANALYZE] {line}", end="")
    analyze_proc.wait()
    if analyze_proc.returncode != 0:
        raise RuntimeError(f"Analyze failed with code {analyze_proc.returncode}")

    # horovodrun 학습 실행
    retrain_save = save_dir / f"plain_training_epochs{epochs}_init-{init}"
    os.makedirs(retrain_save, exist_ok=True)

    train_cmd = [
        "horovodrun",
        "-np",
        str(world_size),
        "python",
        str(train_script),
        "--dataset",
        "imagenet",
        "--num_classes",
        "1000",
        "--dist_mode",
        "single",
        "--workers_per_gpu",
        str(num_workers),
        "--input_image_size",
        str(resolution),
        "--epochs",
        str(epochs),
        "--warmup",
        "5",
        "--optimizer",
        "sgd",
        "--bn_momentum",
        "0.01",
        "--wd",
        "4e-5",
        "--nesterov",
        "--weight_init",
        init,
        "--label_smoothing",
        "--lr_per_256",
        "0.4",
        "--target_lr_per_256",
        "0.0",
        "--lr_mode",
        "cosine",
        "--arch",
        "Masternet.py:MasterNet",
        "--plainnet_struct_txt",
        str(plainnet_txt),
        "--use_se",
        "--target_downsample_ratio",
        "16",
        "--batch_size_per_gpu",
        str(batch_size_per_gpu),
        "--save_dir",
        str(retrain_save),
        "--world-size",
        str(world_size),
        "--dist_mode",
        "horovod",
    ]

    print(f"\n[Running Retrain] {' '.join(train_cmd)}")

    train_proc = subprocess.Popen(
        train_cmd,
        cwd=str(engine_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in train_proc.stdout:
        print(f"[MBV2-TRAIN] {line}", end="")

    train_proc.wait()
    if train_proc.returncode != 0:
        raise RuntimeError(f"Training failed with code {train_proc.returncode}")

    print(f"\n[✅ Completed Retrain] → {retrain_save}")
    return retrain_save