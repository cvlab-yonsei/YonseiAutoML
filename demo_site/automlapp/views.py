import sys, os
sys.path.append(os.path.abspath("/data2/hyunju/data/TempAutoML"))

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
import io, sys, threading
from ysautoml.data.fyi import run_dsa
import subprocess

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
    env = os.environ.copy()
    env["PYTHONPATH"] = "/data2/hyunju/data/TempAutoML"  # ysautoml 루트 경로

    cmd = [
        "python", "-u", "-c",
        (
            "from ysautoml.data.fyi import run_dsa; "
            "run_dsa(dataset='CIFAR10', model='ConvNet', ipc=10, device='0')"
        )
    ]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env
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