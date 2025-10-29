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
