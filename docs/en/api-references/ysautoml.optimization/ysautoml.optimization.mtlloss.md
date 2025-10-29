# ysautoml.optimization.mtl

## Funtional Modules

* `ysautoml.optimization.mtl.examples.nyusp.train_mtl_nyusp`&#x20;

#### ysautoml.optimization.mtl.examples.nyusp.train\_mtl\_nyusp

> ysautoml.optimization.mtl.examples.nyusp.train\_mtl\_nyusp(\*\***kwargs**)

Run **Multi-Task Learning (MTL)** training on the **NYUv2** dataset using the **LibMTL** framework integrated within YSAutoML.\
This function reproduces the original `run.sh` script from `Multi-Task-Learning/examples/nyusp`, allowing you to launch MTL experiments (e.g., GeMTL, GradNorm, UW, DWA, etc.) directly through the YSAutoML API.

> Parameters

* `gpu_id` (`int`, default `0`): Index of the GPU to use for training (e.g., `0`, `1`, ...).
* `seed` (`int`, default `0`): Random seed for reproducibility of task sampling, optimizer initialization, and dataset splits.
* `weighting` (`str`, default `"GeMTL"`): Multi-task weighting strategy.\
  Available options include: `Arithmetic`, `GLS`, `UW`, `DWA`, `RLW`, `GradNorm`, `SI`, `IMTL_L`, `LSBwD`, `LSBwoD`, `AMTL`, `GeMTL`.
* `arch` (`str`, default `"HPS"`): Model architecture type for task-specific decoders and shared backbone.\
  Supported architectures include: `HPS`, `Cross_stitch`, `MTAN`, `CGC`, `PLE`, `MMoE`, `DSelect_k`, `DIY`, `LTB`.
* `dataset_path` (`str`, default `"/dataset/nyuv2"`): Root path of the NYUv2 dataset containing images and labels.
* `scheduler` (`str`, default `"step"`): Learning rate scheduling policy for the optimizer.\
  Common options: `"step"`, `"cos"`, `"exp"`.
* `mode` (`str`, default `"train"`): Operation mode for the run.
  * `"train"`: Train model with train/val/test splits
  * `"test"`: Evaluate a pretrained model only
* `save_dir` (`str`, default `"./logs/nyusp_exp1"`): Directory where experiment logs and checkpoints will be saved.



> Returns

* `None` :  All logs, TensorBoard events, and model checkpoints are saved automatically in `save_dir`.
  * Training logs: `${save_dir}/train.log`
  * TensorBoard events: `${save_dir}/events.out.tfevents.*`
  * Model checkpoints: `${save_dir}/checkpoints/epoch_xx.pth`



#### Examples

```python
from ysautoml.optimization.mtl.examples.nyusp import train_mtl_nyusp

train_mtl_nyusp(
    gpu_id=0,
    seed=42,
    weighting="GeMTL",
    arch="HPS",
    dataset_path="/dataset/nyuv2",
    scheduler="step",
    mode="train",
    save_dir="./logs/nyusp_exp1"
)
```
