# PyTorch implementation of LBT

This is an official implementation of the paper "Toward INT4 Fixed-Point Training via Exploring
Quantization Error for Gradients", accepted to ECCV 2024.

For more information, checkout the project site [[website](https://cvlab.yonsei.ac.kr/projects/LBT/)].

# Getting started 

## Dependencies
* Python >= 3.6
* PyTorch >= 1.8.0


## Training & Evaluation
You can adjust the bit-widths of forward and backward passes in models/modules.py.

To start training, run:
```bash
 python train.py --config configs/resnet20_cifar100.yml
```

## Citation
```
@inproceedings{kim2024toward,
    author={Kim, Dohyung  and Lee, Junghyup and Jeon, Jeimin and Moon, Jaehyeon and Ham, Bumsub},
    title={Toward INT4 Fixed-Point Training via Exploring Quantization Error for Gradients},
    booktitle={European Conference on Computer Vision},
    year={2024},
}
```
---
## Credit
* ResNet-20 model: [[ResNet on CIFAR100](https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py)]
* Quantized modules: [[DSQ](https://github.com/ricky40403/DSQ/blob/master/DSQConv.py#L18)]