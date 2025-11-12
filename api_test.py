#######################

# ## ysautoml.network.fewshot 연습 ##
# from ysautoml.network.fewshot.mobilenet import train_supernet
# import torch
# print("CUDA device count:", torch.cuda.device_count())


# from ysautoml.network.fewshot.mobilenet import train_supernet, train_supernet_decom
from ysautoml.network.fewshot.mobilenet import train_supernet

if __name__ == "__main__":
    result = train_supernet(
        tag="exp1",
        seed=42,
        thresholds=(38, 40),
        num_gpus=2,
        max_epoch=2,
        save_path="./save_dir"  # ✅ 호출 위치 기준 경로
    )

from ysautoml.network.fewshot.mobilenet import search_supernet

search_supernet(
    ckpt="val6-2-seed-0",
    seed=0,
    gpu=0,
    run_calib=True,
    save_path="./save_dir"   # 현재 위치 기준으로 ./logs_my_search/logs/... 저장됨 # 절대 경로도 가능 
) # save path 경로 안에 logs, checkpoint 폴더가 있어야 하며 각각 안에 ckpt에 해당하는 파일명의 파일들을 가지고 있어야 함.



# # 2) decom 버전 학습
# result2 = train_supernet_decom(
#     tag="exp002",
#     seed=1,
#     data_path="/dataset/ILSVRC2012",
#     num_gpus=4,
# )

#######################


## ysautoml.data.fyi ##
from ysautoml.data.fyi import run_dsa, run_dm

# Dataset condensation
run_dsa(dataset="CIFAR10", model="ConvNet", ipc=10, device="0")

# ########################

## ysautoml.data.dsbn ##

# batch 분리 학습
from ysautoml.data.dsbn import convert_and_wrap, train_with_dsbn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# 1. 데이터 준비 (source / target 따로 분리)
transform = transforms.Compose([
    transforms.ToTensor(),
])

full_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
len_source = len(full_train) // 2
len_target = len(full_train) - len_source
source_dataset, target_dataset = random_split(full_train, [len_source, len_target])

source_loader = DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=2)
target_loader = DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=2)

# 2. 모델 변환 (BN → DSBN)
model = convert_and_wrap("resnet18_cifar", dataset="CIFAR10", num_classes=10, use_aug=True, device="0")

# 3. 학습 실행 (분리 학습 모드)
result = train_with_dsbn(model, source_loader, target_loader,
                         epochs=2, lr=0.01, mixed_batch=True, device="cuda")

print("Final Accuracy:", result["final_acc"])
print("Number of log entries:", len(result["logs"]))

# state_dict를 저장하고 싶을 때
import torch
torch.save(result["state_dict"], "./logs/dsbn_trained.pth")

################

# # batch 섞어서 학습
# from ysautoml.data.dsbn import convert_and_wrap, train_with_dsbn_api

# model = convert_and_wrap("resnet18_cifar", dataset="CIFAR10", num_classes=10)

# # mixed_loader (소스+타깃 반반 섞어 배치 구성)
# train_with_dsbn_api(model, mixed_loader,
#                     epochs=10, lr=0.1, mixed_batch=True, device="cuda")


####################################


# from ysautoml.optimization import fxp_quantize
# from ysautoml.network import run_fewshot_nas

# # Quantization
# fxp_quantize(model, dataset, w_bits=4, a_bits=8, g_bits=8)

# # Few-shot NAS
# search_result = run_fewshot_nas(search_space="vit.yaml", dataset="CIFAR100")


# from ysautoml import run_dsa, run_dm

# run_dsa(
#     dataset="CIFAR10",
#     model="ConvNet",
#     ipc=10,
#     dsa_strategy="color_crop_cutout_flip_scale_rotate",
#     init="real", lr_img=1.0, num_exp=5, num_eval=5,
#     run_name="DSAFYI", run_tags="CIFAR10_10IPC", device="0", eval_mode="M",
# )

# run_dm(
#     dataset="CIFAR10",
#     model="ConvNet",
#     ipc=10,
#     dsa_strategy="color_crop_cutout_flip_scale_rotate",
#     init="real", lr_img=1.0, num_exp=5, num_eval=5,
#     run_name="DMFYI", run_tags="CIFAR10_10IPC", device="1", eval_mode="M",
# )
