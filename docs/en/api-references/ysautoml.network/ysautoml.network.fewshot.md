# ysautoml.network.fewshot

## Funtional Modules

* `ysautoml.network.fewshot.mobilenet`

### ysautoml.network.fewshot.mobilenet

#### ysautoml.network.fewshot.mobilenet.train\_supernet

> ysautoml.network.fewshot.mobilenet.train\_supernet(\*\***kwargs)**

Train a shared-weight _SuperNet_ for few-shot MobileNet search using distributed or single-GPU execution.

> Parameters

* `tag` (`str`): Experiment name for log and checkpoint files.
* `seed` (`int`, default `-1`): Random seed for reproducibility.
* `thresholds` (`tuple`, default `(38, 40)`): FLOPs-based sampling thresholds.
* `data_path` (`str`, default `"/dataset/ILSVRC2012"`): Dataset root directory.
* `save_path` (`str`, default `"./SuperNet"`): Directory to store logs and checkpoints.\
  &#xNAN;_&#x41;utomatically resolved relative to the caller’s working directory._
* `search_space` (`str`, default `"proxyless"`): Search space type (`proxyless`, `spos`, etc.).
* `num_gpus` (`int`, default `2`): Number of GPUs to use for training.
* `workers` (`int`, default `4`): Number of dataloader workers.
* `max_epoch` (`int`, default `120`): Total training epochs.
* `train_batch_size` (`int`, default `1024`): Training batch size.
* `test_batch_size` (`int`, default `256`): Validation batch size.
* `learning_rate` (`float`, default `0.12`): Initial learning rate.
* `momentum` (`float`, default `0.9`): SGD momentum factor.
* `weight_decay` (`float`, default `4e-5`): L2 regularization weight.
* `lr_schedule_type` (`str`, default `"cosine"`): Learning rate scheduling strategy.
* `warmup` (`bool`, default `False`): Enable LR warmup.
* `log_to_tb` (`bool`, default `True`): Write TensorBoard logs (`./tb_logs/supernet/{tag}`).



> Returns

* `None` :  Creates `logs/{tag}-seed-{seed}.txt` and `checkpoint/{tag}-seed-{seed}.pt`\
  under the specified `save_path`.



#### Examples

```python
from ysautoml.network.fewshot.mobilenet import train_supernet

train_supernet(
    tag="exp1",
    seed=42,
    thresholds=(38, 40),
    num_gpus=2,
    max_epoch=2,
    save_path="./save_dir"  # resolved relative to current file
)
```

***

#### ysautoml.network.fewshot.mobilenet.search\_supernet

> ysautoml.network.fewshot.mobilenet.search\_supernet(\*\***kwargs)**

Run evolutionary architecture search on a pre-trained MobileNet SuperNet checkpoint.

> Parameters

* `ckpt` (`str`): Checkpoint name to load (e.g., `"baseline0-seed-0"`).
* `seed` (`int`, default `123`): Random seed.
* `gpu` (`int`, default `0`): GPU index.
* `data_path` (`str`, default `"/dataset/ILSVRC2012"`): Dataset root path.
* `save_path` (`str`, default `"./Search"`): Directory for search results.\
  &#xNAN;_&#x41;utomatically resolved relative to the caller’s working directory._
* `search_space` (`str`, default `"proxyless"`): Search space type.
* `workers` (`int`, default `4`): Number of dataloader workers.
* `run_calib` (`bool`, default `False`): Enable BatchNorm calibration before evaluation.



> Returns

* `None` : Evolutionary search results are logged and stored under `save_path`:
  * `logs/ER-{ckpt}-runseed-{seed}.txt`
  * `checkpoint/ER-{ckpt}-runseed-{seed}.pt`



#### Examples

```python
from ysautoml.network.fewshot.mobilenet import search_supernet

search_supernet(
    ckpt="val6-2-seed-0",
    seed=0,
    gpu=0,
    run_calib=True,
    save_path="./save_dir"  # current working directory based
)
```
