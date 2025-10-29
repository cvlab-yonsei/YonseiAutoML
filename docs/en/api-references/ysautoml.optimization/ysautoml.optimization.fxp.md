# ysautoml.optimization.fxp

## Funtional Modules

* `ysautoml.optimization.fxp.train_fxp`

#### ysautoml.network.oneshot.train\_dynas

> ysautoml.optimization.fxp.train\_fxp(\*\***kwargs)**

Run **Fixed-Point Quantization (FXP)** training for a specified model configuration using the YSAutoML optimization engine.

This function launches a separate training process (`engines/train.py`) based on the given YAML configuration file and manages device, seed, and log directory setup for reproducible quantization experiments.

> Parameters

* `config` (`str`): Path to the YAML configuration file defining dataset, model architecture, optimizer, scheduler, and loss parameters (e.g. `"configs/mobilenet_ori.yml"`).
* `device` (`str`, default `cuda:0`): Target device for training. Accepts standard PyTorch device strings such as `"cpu"`, `"cuda:0"`, `"cuda:1"`, etc.
* `seed` (`int`, default `42`): Random seed for reproducibility of model initialization and data order.
* `save_dir` (`str`, default `./logs/fxp_cifar100`): Directory path to store experiment logs, TensorBoard event files, and checkpoint results.



> Returns

* `None` :  All training outputs are written to disk.
  * Logs: `${save_dir}/` (e.g. `logs/fxp_cifar100/events.out.tfevents.*`)
  * Checkpoints: `./results/.../checkpoint/student_epoch_XXXX.pth`
  * Configuration printout: Displayed in console via `pprint(config)`
  * TensorBoard summaries: Scalars and histograms of losses, activations, and quantization parameters





#### Examples

```python
from ysautoml.optimization.fxp import train_fxp

# Run FXP quantization training using ResNet20 on CIFAR-100
train_fxp(
    config="configs/resnet20_cifar100.yml",
    device="cuda:0",
    seed=0,
    save_dir="./logs/fxp_cifar100"
)

# Run FXP quantization training using MobileNet on ImageNet
train_fxp(
    config="configs/mobilenet_ori.yml",
    device="cuda:0",
    seed=42,
    save_dir="./logs/fxp_mobilenet"
)
```

#### Example Configuration Files

configs/resnet20\_cifar100.yml

```yaml
data:
 num_workers: 4
 pin_memory: False 

student_model:
  name: 'resnet20'
  params:
    pretrained: False
    num_classes: 100
    # num_bit: 2
  pretrain: 
    pretrained: False
    dir: ''

train:
  dir: './results'
  batch_size: 128
  num_epochs: 160
  student_dir: '.cifar100.4bit_1'

eval:
  batch_size: 100

scheduler:
 name: 'multi_step'
 params:
   milestones: [80,120]
   gamman: 0.1

q_scheduler:
 name: 'multi_step'
 params:
   milestones: [80,120]
   gamman: 0.1

optimizer:
  name: sgd
  params:
    lr: 0.1
    momentum: 0.9
    weight_decay: 0.0001

q_optimizer:
  name: adam
  params:
    lr: 0.00001
    weight_decay: 0.0

loss:
  name: 'cross_entropy'
  params:
    reduction: 'mean'

gpu: 1
```



configs/mobilenet\_ori.yml

```yaml
data:
 num_workers: 32
 pin_memory: True

student_model:
  name: 'mobilenet_ori'
  params:
    pretrained: True
    num_classes: 1000
    # num_bit: 4
  pretrain:
    pretrained: False
    dir: './results/cvpr_quant/resnet18_ste_mpq_act.w2a2bit_weight_2.0bit_val_0.1con_0.0osi_STE_from_2bit_express_weight_0.5_soft_T10_float_bitloss_0.0_schedule_mu5_adam/checkpoint/epoch_0029.pth'

train:
  dir: './results/cvpr_quant_resnet18'
  batch_size: 256
  num_epochs: 150
  # student_dir: '.c100.4bit_grad_adam1e-5_naive_threshold1e-2_4bit_quant_ep100_ours_decay10_max_avg_real_20e-3'
  # student_dir: '.c100.4bit_grad_adam1e-5_naive_threshold1e-2_4bit_quant_ep100_ours_decay_alpha_1'
  student_dir: '.fp_8gpu_150ep'
  # student_dir: '.c100.4bit_grad_adam1e-5_naive_threshold1e-2_8bit_quant_ep100_gamma_0.8'

eval:
  batch_size: 500

scheduler:
  name: 'cosine'
  params:
    T_max: 750750
    eta_min: 0.0

q_scheduler:
  name: 'cosine'
  params:
    T_max: 750750
    eta_min: 0.0

# scheduler:
#  name: 'multi_step'
#  params:
#    milestones: [150150,300300,450450]
#    gamman: 0.1

# q_scheduler:
#  name: 'multi_step'
#  params:
#    milestones: [150150,300300,450450]
#    gamman: 0.1

# scheduler:
#  name: 'multi_step'
#  params:
#    milestones: [40545,81090,121635]
#    gamman: 0.1

# q_scheduler:
#  name: 'multi_step'
#  params:
#    milestones: [40545,81090,121635]
#    gamman: 0.1

optimizer:
  name: sgd
  params:
    lr: 0.05
    weight_decay: 0.00004

q_optimizer:
  name: adam
  params:
    lr: 0.0
    weight_decay: 0.0

loss:
  name: 'cross_entropy'
  params:
    reduction: 'mean'

gpu: 0
```

