# Multi-Task-Learning

This is an implementation of exploiting the generalized mean for per-task loss aggregation in multi-task learning.
Our code is mainly based on [LibMTL](https://github.com/median-research-group/LibMTL?tab=readme-ov-file).

---
<details>
  <summary>abstract</summary>
  We address the problem of loss balancing for multi-task learning (MTL), which learns multiple tasks simultaneously.
The loss balancing problem is challenging in MTL since each task has a different loss scale, leading to an imbalance between optimizing task-specific losses.
Many approaches to loss balancing typically exploit a linear combination of individual task losses with designed weights. Although these approaches have shown the effectiveness on balancing tasks, there is still a lack of understanding on the underlying principles of balancing in terms of updating task-specific modules.
We first revisit an aggregation of task-specific losses and then show that generalized mean can be a good aggregation which takes advangetes of existing loss balancing methods with a proper parameter which control an aggregation extent while avoiding limitations. 
</details>

## Getting started

1. Create a virtual environment
   
   ```shell
   conda create -n gemtl python=3.8
   conda activate gemtl
   pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. Clone this repository

3. Install `LibMTL`
   
   ```shell
   cd GeMTL
   pip install -e .
   ```

## Requirements

- Python >= 3.8
- Pytorch >= 1.8.1
- GPU: NVIDIA GeForce RTX 2080 Ti
- Driver Version: 525
- CUDA Version: 12.0
- VRAM Usage: Slightly over 7000MB

```shell
pip install -r requirements.txt
```
Please ensure all dependencies are installed to avoid compatibility issues during execution.

## Dataset

You can download datasets in the following links.
- [NYUv2](https://github.com/lorenmt/mtan)  
- [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html)

## Run
Training and testing codes are in `./examples/{nyusp, office}/main.py`, where you can also customize hyperparameters for your experiments.
You can check the results by running the following command.

```shell
cd ./examples/{nyusp, office}
bash run.sh
```
You can modify run.sh scripts to experiment with different datasets or training configurations.
Detailed logging is provided during training to help monitor performance across tasks.

---
Pretrained models and example outputs will be shared in future updates.

## Reference

Our implementation is developed on the following repositories. Thanks to the contributors!
- [LibMTL](https://github.com/median-research-group/LibMTL?tab=readme-ov-file)  
- [CAGrad](https://github.com/Cranial-XIX/CAGrad)  
- [mtan](https://github.com/lorenmt/mtan)

Feel free to contribute by submitting issues or pull requests to improve the repository.

## License

This repository is released under the [GPL-3.0](./LICENSE) license.
