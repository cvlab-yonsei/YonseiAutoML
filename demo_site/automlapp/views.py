import sys, os, time
sys.path.append(os.path.abspath("/data2/hyunju/data/YonseiAutoML"))

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse, FileResponse
import io, sys, threading, traceback
from ysautoml.data.fyi import run_dsa
from ysautoml.data.dsbn import convert_and_wrap, train_with_dsbn
from ysautoml.network.zeroshot.mobilenetv2 import run_search_zeroshot
import subprocess
from pathlib import Path
from django.views.decorators.csrf import csrf_exempt
import json, io, re, os, contextlib, torch
from torchviz import make_dot
from django.conf import settings
from django.views.decorators.http import require_POST
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from contextlib import redirect_stdout, redirect_stderr

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split





# 로그 캡처용 전역 버퍼
log_buffer = []


# def index(request):
#     return HttpResponse("Hello Django!")
def index(request):
    return render(request, 'automlapp/index.html')

def home(request):
    return render(request, "automlapp/home.html")

def data_utility(request):
    return render(request, "automlapp/data.html")

def data_page(request):
    return render(request, "automlapp/data.html")

def total_dashboard(request):
    return render(request, "automlapp/total.html")


def run_dsa_api(request):
    """
    최종 결과 dict 반환
    """
    if request.method == "POST":
        try:
            params = request.POST.dict()
            result = run_dsa(**params)
            return JsonResponse({"result": result})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request"}, status=400)

def run_dsa_stream(request):
    """
    FYI용 데이터 증류 스트리밍 실행 (절대경로 하드코딩 제거 버전)
    """
    # ✅ 현재 Django 프로젝트의 루트 기준
    project_root = Path(settings.BASE_DIR).resolve().parent
    ysa_root = project_root / "ysautoml"

    # ✅ 환경 변수 설정
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)  # 🔥 자동으로 ysautoml 상위 경로 등록

    cmd = [
        sys.executable, "-u", "-c",
        (
            "from ysautoml.data.fyi import run_dsa; "
            "run_dsa(dataset='CIFAR10', model='ConvNet', ipc=10, device='0')"
        )
    ]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True, env=env
    )

    def stream():
        for line in iter(process.stdout.readline, ''):
            yield f"data: {line.strip()}\n\n"
        process.stdout.close()
        process.wait()

    return StreamingHttpResponse(stream(), content_type="text/event-stream")

def fetch_logs(request):
    return JsonResponse({"logs": log_buffer})

def network_utility(request):
    return render(request, "automlapp/network.html")

def optimization_utility(request):
    return render(request, "automlapp/optimization.html")

def run_automl(request):
    if request.method == 'POST':
        dataset = request.POST.get('dataset')
        model = request.POST.get('model')
        ipc = int(request.POST.get('ipc'))
        device = request.POST.get('device')

        # run_dsa 실행
        try:
            run_dsa(dataset=dataset, model=model, ipc=ipc, device=device)
            result = f"✅ run_dsa 실행 성공! dataset={dataset}, model={model}, ipc={ipc}, device={device}"
        except Exception as e:
            result = f"❌ 실행 실패: {str(e)}"

        return HttpResponse(result)
    else:
        return HttpResponse("잘못된 요청입니다.")

# --- SSE 스트리밍 함수 ---
def run_total_stream(request):
    def event_stream():
        yield "data: [Stage 1] Initializing YSAutoML total pipeline...\n\n"
        time.sleep(1)
        yield "data: [Stage 1] Network search started...\n\n"
        time.sleep(2)
        yield "data: [Stage 1] Search complete. Found best checkpoint: ckpt_best_oneshot.pth\n\n"

        yield "data: [Stage 2] Applying selected data modules (FYI / DSBN)...\n\n"
        time.sleep(2)
        yield "data: [Stage 2] Data processing complete.\n\n"

        yield "data: [Stage 3] Running optimization (Loss Search / MTL)...\n\n"
        time.sleep(3)
        yield "data: [Stage 3] Optimization complete. Final model: final_losssearch.pth\n\n"

        yield "data: ✅ Pipeline finished successfully!\n\n"

    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    return response


# --- 최종 실행 결과 반환 ---
def run_total(request):
    if request.method == "POST":
        model = request.POST.get("model")
        network = request.POST.get("network")
        optimization = request.POST.get("optimization")
        data_methods = request.POST.getlist("data")

        # 실제 실행 로직을 별도의 스레드에서 처리 가능
        # threading.Thread(target=run_pipeline, args=(model, network, optimization, data_methods)).start()

        result = {
            "model": model,
            "network": network,
            "optimization": optimization,
            "data_methods": data_methods,
            "checkpoint": f"final_{optimization}.pth",
            "logs": [
                f"Network search: {network} completed.",
                f"Data applied: {', '.join(data_methods)}",
                f"Optimization: {optimization} done.",
            ],
        }
        return JsonResponse({"result": result})
    else:
        return JsonResponse({"error": "Invalid request method."}, status=400)


@csrf_exempt
def run_total_pipeline(request):
    if request.method == "POST":
        data = json.loads(request.body)
        network = data.get("network")

        if network == "zeroshot":
            gpu = int(data.get("gpu", 0))
            seed = int(data.get("seed", 123))
            metric = data.get("metric", "AZ_NAS")
            population = int(data.get("population_size", 100))
            evo_iter = int(data.get("evolution_max_iter", 100))
            resolution = int(data.get("resolution", 224))
            budget_flops = float(data.get("budget_flops", 1e9))
            max_layers = int(data.get("max_layers", 16))
            batch_size = int(data.get("batch_size", 32))
            data_path = data.get("data_path", "/dataset/ILSVRC2012/")

            def stream():
                yield "🚀 Starting Zero-Shot NAS...\n"

                try:
                    # ✅ subprocess로 별도 프로세스에서 실행 (stdout 즉시 flush)
                    cmd = [
                        sys.executable, "-u", "-c",
                        (
                            "from ysautoml.network.zeroshot.mobilenetv2.api import run_search_zeroshot; "
                            f"run_search_zeroshot("
                            f"gpu={gpu}, seed={seed}, metric='{metric}', "
                            f"population_size={population}, evolution_max_iter={evo_iter}, "
                            f"resolution={resolution}, budget_flops={budget_flops}, "
                            f"max_layers={max_layers}, batch_size={batch_size}, "
                            f"data_path='{data_path}')"
                        )
                    ]

                    process = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, universal_newlines=True
                    )

                    for line in iter(process.stdout.readline, ""):
                        yield line
                        # ✅ 다운로드 경로 감지
                        if "Completed Search & Analyze" in line:
                            match = re.search(r"→ (/.+)", line)
                            if match:
                                best_file = os.path.join(match.group(1).strip(), "best_structure.txt")
                                yield f"\n[DOWNLOAD_READY] {best_file}\n"

                    process.stdout.close()
                    process.wait()

                    yield "\n✅ Completed Zero-Shot NAS.\n"

                except Exception as e:
                    yield f"\n❌ Error: {str(e)}\n"

            return StreamingHttpResponse(stream(), content_type="text/plain")

    return JsonResponse({"error": "Invalid request"}, status=400)


# ✅ 파일 다운로드 뷰
@csrf_exempt
def download_file(request):
    path = request.GET.get("path")
    if path and os.path.exists(path):
        filename = os.path.basename(path)
        return FileResponse(open(path, "rb"), as_attachment=True, filename=filename)
    return JsonResponse({"error": "File not found"}, status=404)



@csrf_exempt
@require_POST
def visualize_model_from_structure(request):
    """
    best_structure.txt 경로를 받아서 모델 그래프를 torchviz로 시각화.
    절대경로 없이 BASE_DIR과 상대경로 기반으로 처리.
    """
    print("hi111")
    print("request.body >>>", request.body)

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    if request.method != "POST":
        print("hi222")
        return JsonResponse({"error": "Invalid request"}, status=400)

    try:
        print("hi333")
        data = json.loads(request.body)
        struct_path = data.get("path")

        # 1️⃣ 구조파일 유효성 검사
        if not struct_path:
            return JsonResponse({"error": "No structure path provided"}, status=400)
        print("11111")
        # 절대경로 or 상대경로 모두 허용
        struct_path = Path(struct_path)
        if not struct_path.is_absolute():
            struct_path = (Path.cwd() / struct_path).resolve()
        print("22222")
        if not struct_path.exists():
            return JsonResponse({"error": f"File not found: {struct_path}"}, status=404)
        print("33333")
        # 2️⃣ 프로젝트 내 ysaautoml 경로 자동 탐색
        base_dir = Path(settings.BASE_DIR).resolve()
        project_root = base_dir.parent
        ysa_path = None
        for subdir in ["ysautoml", "YSAutoML"]:
            candidate = project_root / subdir
            if candidate.exists():
                ysa_path = candidate
                break
        print("44444")
        if ysa_path is None:
            return JsonResponse({"error": "Cannot locate ysautoml directory."}, status=500)
        
        print("5555")

        # sys.path.append(str(ysa_path / "network" / "zeroshot" / "mobilenetv2"))
        
        print("hi444 - importing model loader")

        from ysautoml.network.zeroshot.mobilenetv2.engines.ImageNet_MBV2 import ModelLoader

        # -----------------------------------------
        # opt, argv 직접 구성 (run_retrain_zeroshot 기반)
        # -----------------------------------------
        argv = [
            "--dataset", "imagenet",
            "--num_classes", "1000",
            "--input_image_size", "224",
            "--arch", "Masternet.py:MasterNet",
            "--plainnet_struct_txt", str(struct_path),
            "--use_se",
            "--target_downsample_ratio", "16",
            "--batch_size_per_gpu", "64",
        ]

        input_image_size = 224

        class DummyOpt:
            pass

        opt = DummyOpt()
        opt.dataset = "imagenet"
        opt.num_classes = 1000
        opt.input_image_size = input_image_size
        opt.arch = "Masternet.py:MasterNet"
        opt.plainnet_struct_txt = str(struct_path)
        opt.use_se = True
        opt.target_downsample_ratio = 16
        opt.batch_size_per_gpu = 64
        opt.save_dir = str(base_dir / "static" / "visuals")

        # ✅ ModelLoader 내부 참조 대비
        opt.pretrained = False
        opt.bn_momentum = 0.01
        opt.wd = 4e-5
        opt.weight_init = "custom_kaiming"
        opt.nesterov = True
        opt.world_size = 1
        opt.dist_mode = "single"
        opt.workers_per_gpu = 4
        opt.optimizer = "sgd"
        opt.lr_per_256 = 0.4
        opt.target_lr_per_256 = 0.0
        opt.lr_mode = "cosine"
        opt.use_label_smoothing = True


        print(f"[INFO] Building model from structure: {struct_path}", flush=True)
        model = ModelLoader.get_model(opt, argv)
        x = torch.randn(1, 3, input_image_size, input_image_size, requires_grad=True)

         # ✅ eval mode 강제 적용
        model.eval()
        for m in model.modules():
            if hasattr(m, "training"):
                m.train(False)

        y_pred = model(x)
        if isinstance(y_pred, dict):
            y_pred = y_pred.get("out", list(y_pred.values())[0])

        # ✅ torchviz 시각화
        dot = make_dot(
            y_pred,
            params=dict(model.named_parameters()),
            # show_attrs=True,
            # show_saved=True
        )
        output_dir = base_dir / "static" / "visuals"
        output_dir.mkdir(parents=True, exist_ok=True)

        img_path = output_dir / f"model_graph_{struct_path.stem}.png"
        dot.render(filename=img_path.with_suffix(''), format="png", cleanup=True)

        img_url = f"/static/visuals/{img_path.name}"
        return JsonResponse({"url": img_url})

    except Exception as e:
        err_msg = traceback.format_exc()

        # 🚨 1️⃣ 표준출력 강제 flush
        sys.stderr.write("\n" + "="*80 + "\n")
        sys.stderr.write("🔥 [visualize_model_from_structure ERROR - STDERR]\n")
        sys.stderr.write(err_msg)
        sys.stderr.write("\n" + "="*80 + "\n")
        sys.stderr.flush()

        # 🚨 2️⃣ 표준출력도 함께 강제 flush
        sys.stdout.write("\n" + "="*80 + "\n")
        sys.stdout.write("🔥 [visualize_model_from_structure ERROR - STDOUT]\n")
        sys.stdout.write(err_msg)
        sys.stdout.write("\n" + "="*80 + "\n")
        sys.stdout.flush()

        # 🚨 3️⃣ 임시로 바로 중단시켜서 Django 기본 traceback 출력 유도
        raise e


@csrf_exempt
def run_fxp_training(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    config_path = "/data1/hyunju/code/YonseiAutoML/ysautoml/optimization/fxp/engines/configs/mobilenet_2bit.yml"
    dsbn_flag = request.POST.get("dsbn", "false").lower() == "true"
    fyi_flag  = request.POST.get("fyi", "false").lower() == "true"

    best_struct_path = Path(
        "/data1/hyunju/code/YonseiAutoML/ysautoml/network/zeroshot/mobilenetv2/engines/"
        "ImageNet_MBV2/save_dir/AZ_NAS_flops1G-searchbs32-pop100-iter100-123/best_structure.txt"
    )

    def stream_fxp():
        yield f"[FXP] Config: {config_path}\n"
        yield f"[FXP] DSBN={dsbn_flag}, FYI={fyi_flag}\n"
        yield f"[FXP] Using best structure: {best_struct_path}\n"
        yield "[FXP] Starting FXP training...\n\n"
        sys.stdout.flush()

        # ✅ subprocess 실행 (stdout을 스트리밍으로 바로 읽기)
        cmd = [
            sys.executable, "-u", "-c",
            (
                "from ysautoml.optimization.fxp import train_fxp; "
                "trained_pth = train_fxp("
                f"config='{config_path}', "
                f"device='cuda:0', seed=42, "
                f"save_dir='./logs/fxp_imagenet', "
                f"arch_path='{best_struct_path}', "
                f"dsbn={dsbn_flag}, fyi={fyi_flag}); "
                "print(f'\\n[FXP_DONE] {trained_pth}')"
            )
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        for line in iter(process.stdout.readline, ""):
            yield line
            sys.stdout.write(line)  # 터미널에도 출력
            sys.stdout.flush()

        process.stdout.close()
        process.wait()

        yield "\n✅ FXP training process finished.\n"

    response = StreamingHttpResponse(stream_fxp(), content_type="text/plain")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


# @csrf_exempt
# def run_fxp_training(request):
#     if request.method != "POST":
#         return JsonResponse({"error": "Invalid method"}, status=405)

#     # ✅ 업로드된 config.yml 저장
#     config_file = request.FILES["config_file"]
#     tmp_dir = Path(tempfile.mkdtemp(prefix="fxp_"))
#     config_path = tmp_dir / config_file.name
#     default_storage.save(str(config_path), ContentFile(config_file.read()))

#     dsbn_flag = request.POST.get("dsbn", "false").lower() == "true"
#     fyi_flag = request.POST.get("fyi", "false").lower() == "true"

#     # ✅ zero-shot 과정에서 생성된 best_structure.txt 경로 자동 참조
#     best_struct_path = Path("/data1/hyunju/code/YonseiAutoML/ysautoml/network/zeroshot/mobilenetv2/engines/ImageNet_MBV2/save_dir/AZ_NAS_flops1G-searchbs32-pop100-iter100-123/best_structure.txt")

#     def stream_fxp():
#         yield f"[FXP] Config: {config_path}\n"
#         yield f"[FXP] DSBN={dsbn_flag}, FYI={fyi_flag}\n"
#         yield f"[FXP] Using best structure: {best_struct_path}\n"
#         try:
#             from ysautoml.optimization.fxp import train_fxp
#             yield "[FXP] Starting FXP training...\n"

#             trained_pth = train_fxp(
#                 config=str(config_path),
#                 device="cuda:0",
#                 seed=42,
#                 save_dir="./logs/fxp_imagenet",
#                 arch_path=str(best_struct_path),
#                 dsbn=dsbn_flag,
#                 fyi=fyi_flag,
#             )

#             if trained_pth:
#                 yield f"[FXP_DONE] {trained_pth}\n"
#             else:
#                 yield "[FXP] Training finished but no .pth found.\n"
#         except Exception as e:
#             yield f"[FXP_ERROR] {e}\n"

#     return StreamingHttpResponse(stream_fxp(), content_type="text/plain")

# 공통 결과 저장 경로 (SSE 실행 중에 결과를 파일로 떨궈두고 /api에서 읽음)
_TMP_DIR = Path(settings.BASE_DIR).resolve() / "static" / "tmp"
_TMP_DIR.mkdir(parents=True, exist_ok=True)
_DSBN_CONVERT_JSON = _TMP_DIR / "dsbn_convert_result.json"
_DSBN_TRAIN_JSON   = _TMP_DIR / "dsbn_train_result.json"

# -----------------------------
# DSBN Convert - SSE Stream
# -----------------------------
def dsbn_convert_stream(request):
    """
    convert_and_wrap 실행 로그를 SSE로 전달.
    convert는 내부적으로 크게 프린트가 많지 않으니, 여기서 수동으로 단계 로그를 보냄.
    """
    # GET 파라미터
    model_or_name = request.GET.get("model_or_name", "").strip() or None
    dataset = request.GET.get("dataset", "CIFAR10")
    num_classes = int(request.GET.get("num_classes", 10))
    use_aug = request.GET.get("use_aug", "false").lower() == "true"
    mode_str = request.GET.get("mode", "").strip()
    mode = int(mode_str) if mode_str.isdigit() else None
    device = request.GET.get("device", "0")
    export_path = request.GET.get("export_path", "").strip() or None

    def stream():
        try:
            yield f"data: [DSBN-CONVERT] Starting... dataset={dataset}, num_classes={num_classes}, use_aug={use_aug}, mode={mode}, device={device}\n\n"

            # 실제 변환
            model = convert_and_wrap(
                model_or_name=model_or_name or "resnet18_cifar",
                dataset=dataset,
                num_classes=num_classes,
                use_aug=use_aug,
                mode=mode,
                device=device,
                export_path=export_path
            )

            yield f"data: [DSBN-CONVERT] Model converted to DSBN. Mode set. ({'inferred from use_aug' if mode is None else f'mode={mode}'})\n\n"

            if export_path:
                yield f"data: [DSBN-CONVERT] state_dict saved to: {export_path}\n\n"

            # 결과 JSON 떨구기
            result = {
                "model": model_or_name or "resnet18_cifar",
                "dataset": dataset,
                "num_classes": num_classes,
                "mode": mode_str or None,
                "exported_path": export_path,
            }
            _DSBN_CONVERT_JSON.write_text(json.dumps(result), encoding="utf-8")
            yield "data: [DONE]\n\n"

        except Exception as e:
            err = f"[ERROR] {str(e)}"
            yield f"data: {err}\n\n"

    resp = StreamingHttpResponse(stream(), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    return resp

# -----------------------------
# DSBN Convert - Final JSON
# -----------------------------
@csrf_exempt
def dsbn_convert_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)
    try:
        if not _DSBN_CONVERT_JSON.exists():
            return JsonResponse({"error": "No convert result found."}, status=404)
        data = json.loads(_DSBN_CONVERT_JSON.read_text(encoding="utf-8"))
        return JsonResponse({"result": data})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# -----------------------------
# DSBN Train - SSE Stream
# -----------------------------
def dsbn_train_stream(request):
    dataset = request.GET.get("dataset", "CIFAR10")
    batch_size = int(request.GET.get("batch_size", 128))
    epochs = int(request.GET.get("epochs", 1))
    lr = float(request.GET.get("lr", 0.01))
    mixed_batch = request.GET.get("mixed_batch", "false").lower() == "true"
    device = request.GET.get("device", "cuda")

    def stream():
        try:
            yield f"data: [DSBN-TRAIN] Preparing dataset {dataset}...\n\n"
            transform = transforms.Compose([transforms.ToTensor()])

            if dataset.upper() == "CIFAR100":
                full_train = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
                num_classes = 100
            else:
                full_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
                num_classes = 10

            len_source = len(full_train) // 2
            len_target = len(full_train) - len_source
            source_dataset, target_dataset = random_split(full_train, [len_source, len_target])

            source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

            yield f"data: [DSBN-TRAIN] Converting model to DSBN...\n\n"
            model = convert_and_wrap(
                model_or_name="resnet18_cifar",
                dataset=dataset,
                num_classes=num_classes,
                use_aug=mixed_batch,
                mode=(3 if mixed_batch else 1),
                device="0" if device.startswith("cuda") else device,
            )

            yield f"data: [DSBN-TRAIN] Start training... epochs={epochs}, lr={lr}, mixed_batch={mixed_batch}\n\n"

            # === 핵심 수정 ===
            result = train_with_dsbn(
                model,
                source_loader=source_loader if not mixed_batch else source_loader,
                target_loader=None if mixed_batch else target_loader,
                epochs=epochs,
                lr=lr,
                mixed_batch=mixed_batch,
                device=device,
            )
            # === 여기까지 ===

            yield f"data: [DSBN-TRAIN] Training finished. Collecting logs...\n\n"

            logs = result.get("logs", [])
            final_acc = result.get("final_acc", None)
            state_dict = result.get("state_dict", None)

            # logs 리스트를 하나씩 yield 해줌
            if logs:
                for entry in logs[:50]:  # 너무 길면 50개까지만
                    yield f"data: {entry}\n\n"
                    time.sleep(0.01)

            if final_acc is not None:
                yield f"data: [DSBN-TRAIN] Final Acc = {final_acc}\n\n"

            state_path = None
            if state_dict:
                out_path = _TMP_DIR / "dsbn_trained.pth"
                torch.save(state_dict, out_path)
                state_path = str(out_path)
                yield f"data: [DSBN-TRAIN] Saved model → {state_path}\n\n"

            final_result = {
                "dataset": dataset,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "mixed_batch": mixed_batch,
                "final_acc": final_acc,
                "state_dict_path": state_path,
            }
            _DSBN_TRAIN_JSON.write_text(json.dumps(final_result), encoding="utf-8")

            yield f"data: [DSBN-TRAIN] ✅ Training complete.\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            err = traceback.format_exc()
            yield f"data: [ERROR] {e}\n\n"
            yield f"data: {err}\n\n"

    resp = StreamingHttpResponse(stream(), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    return resp


# -----------------------------
# DSBN Train - Final JSON
# -----------------------------
@csrf_exempt
def dsbn_train_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)
    try:
        if not _DSBN_TRAIN_JSON.exists():
            return JsonResponse({"error": "No train result found."}, status=404)
        data = json.loads(_DSBN_TRAIN_JSON.read_text(encoding="utf-8"))
        return JsonResponse({"result": data})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)