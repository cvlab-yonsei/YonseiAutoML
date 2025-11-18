import argparse

def make_parser():
    p = argparse.ArgumentParser(description="DSBN (Domain-Specific BatchNorm) Utilities")

    # number of model/dataset/class 
    p.add_argument("--model", type=str, default="resnet18_cifar", help="model name")
    p.add_argument("--num_classes", type=int, default=10, help="number of classes")
    p.add_argument("--dataset", type=str, default="CIFAR10", help="dataset name")

    # training option
    p.add_argument("--epochs", type=int, default=1, help="training epochs")
    p.add_argument("--lr", type=float, default=0.1, help="learning rate")
    p.add_argument("--batch_size", type=int, default=128, help="batch size")
    p.add_argument("--mixed_batch", action="store_true",
                   help="if set, use mixed batch mode (source+target in same batch)")

    # DSBN 동작 옵션
    p.add_argument("--use_aug", action="store_true", help="use target/aug branch")
    p.add_argument("--mode", type=int, default=None, help="DSBN mode: 1(S), 2(T), 3(split half)")
    p.add_argument("--device", type=str, default="0", help="CUDA_VISIBLE_DEVICES")

    # export
    p.add_argument("--export_path", type=str, default=None, help="path to save converted model state_dict")

    return p
