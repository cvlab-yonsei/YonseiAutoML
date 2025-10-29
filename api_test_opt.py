from ysautoml.optimization.fxp import train_fxp

if __name__ == "__main__":
    train_fxp(
        # config="configs/resnet20_cifar100.yml",
        config="configs/mobilenet_ori.yml",
        device="cuda:0",
        seed=42,
        save_dir="./logs/fxp_cifar100"
    )
