# ysautoml.optimization.losssearch

## Funtional Modules

* `ysautoml.optimization.losssearch.train_losssearch`&#x20;
* `ysautoml.optimization.losssearch.custom_loss`&#x20;

Run Loss Function Search (LFS) using a custom, learnable loss function integrated into the YSAutoML optimization system.\
This module supports two main functionalities:

1. Launching a complete end-to-end training pipeline (`train_losssearch`) where both model and loss function are jointly optimized.
2. Directly accessing and using the learnable criterion (`custom_loss`) in custom training or evaluation scripts.

#### ysautoml.optimization.losssearch.train\_losssearch

> ysautoml.optimization.losssearch.train\_losssearch(\*\***kwargs**)

Launching a complete end-to-end training pipeline (`train_losssearch`) where both model and loss function are jointly optimized.

> Parameters

* `epochs` (`int`, default `100`): Total number of training epochs.
* `lr_model` (`float`, default `0.1`): Learning rate for the model optimizer.
* `lr_loss` (`float`, default `0.0001`): Learning rate for the optimizer that updates the custom loss parameters.
* `momentum` (`float`, default `0.9`): Momentum factor for model and loss optimizers.
* `weight_decay` (`float`, default `0.0005`): Weight decay coefficient for L2 regularization in model optimization.
* `save_dir` (`str`, default `"./logs/losssearch"`): Directory to save logs, checkpoints, and learned loss parameters.
* `device` (`str`, default `"cuda:0"`): Device identifier (e.g., `"cuda:0"`, `"cuda:1"`, or `"cpu"`).



> Returns

* `None` :  All training logs, learned loss parameter values, and model checkpoints are saved under the specified `save_dir`.
  * Logs: `${save_dir}/training.log`
  * TensorBoard events: `${save_dir}/events.out.tfevents.*`
  * Checkpoints: `${save_dir}/model_epoch_XX.pth`
  * Learned loss parameters: saved in model state dict as `criterion.theta`



#### Examples

```python
from ysautoml.optimization.losssearch import train_losssearch

train_losssearch(
    epochs=50,
    lr_model=0.05,
    lr_loss=0.0005,
    momentum=0.9,
    weight_decay=0.0001,
    save_dir="./logs/losssearch_exp1",
    device="cuda:0"
)
```

***

#### ysautoml.optimization.losssearch.custom\_loss

> ysautoml.optimization.losssearch.custom\_loss( )

Directly accessing and using the learnable criterion (`custom_loss`) in custom training or evaluation scripts.

> Parameters

* `None`&#x20;

#### Examples

```python
from ysautoml.optimization.losssearch import custom_loss
import torch

criterion = custom_loss().cuda()
preds = torch.randn(8, 10).cuda()
targets = torch.randint(0, 10, (8,)).cuda()
loss = criterion(preds, targets)
loss.backward()

```
