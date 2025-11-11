from ysautoml.optimization.fxp import train_fxp

if __name__ == "__main__":
    train_fxp(
        # config="configs/resnet20_cifar100.yml",
        # config="configs/mobilenet_ori.yml",
        config="configs/mobilenet_2bit.yml",
        device="cuda:0",
        seed=42,
        # save_dir="./logs/fxp_cifar100",
        save_dir="./logs/fxp_imagenet",
        arch_path="/data1/hyunju/code/YonseiAutoML/ysautoml/network/zeroshot/mobilenetv2/engines/ImageNet_MBV2/save_dir/AZ_NAS_flops1G-searchbs32-pop100-iter100-123/best_structure.txt"
    )

# ###############

# from ysautoml.optimization.losssearch import train_losssearch

# train_losssearch(
#     epochs=50,
#     lr_model=0.05,
#     lr_loss=0.0005,
#     momentum=0.9,
#     weight_decay=0.0001,
#     save_dir="./logs/losssearch_exp1",
#     device="cuda:0"
# )


# ###############

# from ysautoml.optimization.losssearch import custom_loss
# import torch

# criterion = custom_loss().cuda()
# preds = torch.randn(8, 10).cuda()
# targets = torch.randint(0, 10, (8,)).cuda()
# loss = criterion(preds, targets)
# loss.backward()

# ##############

# from ysautoml.optimization.mtl.examples.nyusp import train_mtl_nyusp

# train_mtl_nyusp(
#     gpu_id=0,
#     seed=42,
#     weighting="GeMTL",
#     arch="HPS",
#     dataset_path="/dataset/nyuv2",
#     scheduler="step",
#     mode="train",
#     save_dir="./logs/nyusp_exp1"
# )
