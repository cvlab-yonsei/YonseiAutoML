# ysautoml.data.fyi

## funtional

### ysautoml.data.fyi.run\_dsa

> ysautoml.data.fyi.run\_dsa(\*\*kwargs)

Run dataset condensation using **Differentiable Siamese Augmentation (DSA)**.

> Parameters

* `dataset` (str): Dataset name (e.g., `'CIFAR10'`, `'CIFAR100'`).
* `model` (str): Backbone model (e.g., `'ConvNet'`).
* `ipc` (int): Images per class.
* `eval_mode` (str): Evaluation mode. Options:
  * `'S'`: same as training model
  * `'M'`: multi architectures
  * `'W'`: net width
  * `'D'`: net depth
  * `'A'`: activation function
  * `'P'`: pooling layer
  * `'N'`: normalization layer
* `num_exp` (int): Number of experiments.
* `num_eval` (int): Number of evaluation models.
* `epoch_eval_train` (int): Epochs for evaluation training.
* `Iteration` (int, default=1000): Training iterations.
* `lr_img` (float): Learning rate for synthetic images.
* `lr_net` (float): Learning rate for network parameters.
* `batch_real` (int): Batch size for real data.
* `batch_train` (int): Batch size for synthetic training.
* `init` (str): Initialization mode. Options: `'noise'`, `'real'`.
* `dsa_strategy` (str): Augmentation strategy (comma-separated).
* `data_path` (str): Path to dataset root.
* `device` (str): Device ID, e.g. `'0'`.
* `run_name` (str): Experiment name.
* `run_tags` (str, optional): Tags for logging.



> Returns

* `dict`: Containing results of all experiments:
  * `save_path` (str): Directory of logs and checkpoints.
  * `accs_all_exps` (dict): Recorded accuracies for each evaluation model.
  * `eval_pool` (list): Evaluation model pool.
  * `num_exp` (int): Number of experiments.



#### Examples

```python
from ysautoml.data.fyi import run_dsa

run_dsa(
    dataset="CIFAR10",
    model="ConvNet",
    ipc=10,
    dsa_strategy="color_crop_cutout_flip_scale_rotate",
    init="real", lr_img=1.0, num_exp=5, num_eval=5,
    run_name="DSAFYI", run_tags="CIFAR10_10IPC", device="0", eval_mode="M",
)
```

***

### ysautoml.data.fyi.run\_dm

> ysautoml.data.fyi.run\_dm(\*\*kwargs)

Run dataset condensation using **Distribution Matching (DM)**.

> Parameters

* (same as `run_dsa`, except default `Iteration=20000`)

> Returns

* `dict`: Same format as `run_dsa`.



#### Examples

```python
from ysautoml.data.fyi import run_dm

run_dm(
    dataset="CIFAR10",
    model="ConvNet",
    ipc=10,
    dsa_strategy="color_crop_cutout_flip_scale_rotate",
    init="real", lr_img=1.0, num_exp=5, num_eval=5,
    run_name="DMFYI", run_tags="CIFAR10_10IPC", device="1", eval_mode="M",
)
```
