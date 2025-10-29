# ysautoml.network.zeroshot

## Funtional Modules

* `ysautoml.network.zeroshot.autoformer`
* `ysautoml.network.zeroshot.mobilenetv2`

### ysautoml.network.zeroshot.autoformer

#### ysautoml.network.zeroshot.autoformer.run\_search\_zeroshot

> ysautoml.network.zeroshot.autoformer.run\_search\_zeroshot(\*\*kwargs)

Run zero-shot architecture search using **AutoFormer** on ImageNet.

> Parameters

* `param_limits` (float): Maximum parameter count limit (e.g., 6, 23, 54 for Tiny/Small/Base).
* `min_param_limits` (float): Minimum parameter count limit.
* `cfg` (str): Path or name of YAML search space (e.g., `'space-T.yaml'`).
* `output_dir` (str): Directory to save search results.
* `data_path` (str, default `'/dataset/ILSVRC2012'`): Dataset root path.
* `population_num` (int, default `10000`): Number of architectures to sample.
* `seed` (int, default `123`): Random seed.
* `gp` (bool, default `True`): Enable Gaussian process metric.
* `relative_position` (bool, default `True`): Enable relative positional embedding.
* `change_qkv` (bool, default `True`): Use QKV reparameterization.
* `dist_eval` (bool, default `True`): Enable distributed evaluation.



> Returns

* `None` : Logs and search results (including `best_arch.yaml` and `search.log`) are saved under `output_dir`.



#### Examples

```python
from ysautoml.network.zeroshot.autoformer import run_search_zeroshot

# Tiny Search
run_search_zeroshot(
    param_limits=6,
    min_param_limits=4,
    cfg="space-T.yaml",
    output_dir="./OUTPUT/search/AZ-NAS/Tiny"
)
```

***

#### ysautoml.network.zeroshot.autoformer.run\_retrain\_zeroshot

> ysautoml.network.zeroshot.autoformer.run\_retrain\_zeroshot(\*\*kwargs)

Retrain the best subnet architecture found by AZ-NAS using AutoFormer.

> Parameters

* `cfg` (str): YAML configuration file path (absolute or relative).
* `output_dir` (str): Output directory for retraining results.
* `data_path` (str, default `'/dataset/ILSVRC2012'`): Dataset root path.
* `epochs` (int, default `500`): Number of training epochs.
* `warmup_epochs` (int, default `20`): Warm-up epochs.
* `batch_size` (int, default `256`): Batch size per GPU.
* `model_type` (str, default `'AUTOFORMER'`): Model type to train.
* `mode` (str, default `'retrain'`): Training mode.
* `relative_position` (bool): Use relative positional embedding.
* `change_qkv` (bool): Use QKV reparameterization.
* `gp` (bool): Enable Gaussian process module.
* `dist_eval` (bool): Enable distributed evaluation.
* `device` (str, default `'0,1,2,3,4,5,6,7'`): Visible CUDA devices.
* `nproc_per_node` (int, default `8`): Number of distributed processes per node.
* `master_port` (int, default `6666`): Master port for distributed training.



> Returns

* `None` : Trained model weights and logs are saved under `output_dir`.



#### Examples

```python
from ysautoml.network.zeroshot.autoformer import run_retrain_zeroshot

run_retrain_zeroshot(
    cfg="./Tiny.yaml",
    output_dir="./OUTPUT/AZ-NAS/Tiny-bs256x8-use_subnet-500ep",
    epochs=500
)
```



***



### ysautoml.network.zeroshot.mobilenetv2

#### ysautoml.network.zeroshot.mobilenetv2.run\_search\_zeroshot

> ysautoml.network.zeroshot.mobilenetv2.run\_search\_zeroshot(\*\*kwargs)

Run zero-shot evolution search for MobileNetV2 variants (AZ-NAS).

> Parameters

* `gpu` (int, default `0`): GPU ID to use.
* `seed` (int, default `123`): Random seed.
* `metric` (str, default `'AZ_NAS'`): Zero-shot score metric.
* `population_size` (int, default `1024`): Population size for evolution search.
* `evolution_max_iter` (int, default `1e5`): Maximum iterations for evolution search.
* `resolution` (int, default `224`): Input image resolution.
* `budget_flops` (float, default `1e9`): FLOPs constraint.
* `max_layers` (int, default `16`): Maximum number of layers.
* `batch_size` (int, default `64`): Batch size.
* `data_path` (str): Path to ImageNet dataset.
* `num_classes` (int, default `1000`): Number of classes.
* `search_space` (str, default `'SearchSpace/search_space_IDW_fixfc.py'`): Search space file.



> Returns

* `None` – Outputs best architecture (`best_structure.txt`) and FLOPs/params summary in `save_dir`.



#### Examples

```python
from ysautoml.network.zeroshot.mobilenetv2 import run_search_zeroshot

# Default 1G search
run_search_zeroshot(
    gpu=0,
    seed=123,
    metric="AZ_NAS",
    population_size=1024,
    evolution_max_iter=int(1e5),
    resolution=224,
    budget_flops=1e9,
    max_layers=16,
    batch_size=64,
    data_path="/dataset/ILSVRC2012/",
)

# Small (450M)
run_search_zeroshot(
    gpu=0,
    seed=123,
    budget_flops=450e6,
    max_layers=14,
)

# Medium (600M)
run_search_zeroshot(
    gpu=0,
    seed=123,
    budget_flops=600e6,
    max_layers=14,
)

# Large (1G)
run_search_zeroshot(
    gpu=0,
    seed=123,
    budget_flops=1000e6,
    max_layers=16,
)

```

***

#### ysautoml.network.zeroshot.mobilenetv2.run\_retrain\_zeroshot

> ysautoml.network.zeroshot.mobilenetv2.run\_retrain\_zeroshot(\*\*kwargs)

Retrain the searched MobileNetV2 architecture using AZ-NAS configuration (via Horovod).

> Parameters

* `gpu_devices` (str, default `'0,1,2,3,4,5,6,7'`): Visible GPU device IDs.
* `metric` (str, default `'AZ_NAS'`): Search metric name.
* `population_size` (int, default `1024`): Population size used in search.
* `evolution_max_iter` (int, default `1e5`): Number of evolution iterations used in search.
* `seed` (int, default `123`): Random seed.
* `num_workers` (int, default `12`): Number of data loader workers.
* `init` (str, default `'custom_kaiming'`): Weight initialization method.
* `epochs` (int, default `150`): Number of training epochs.
* `resolution` (int, default `224`): Input image resolution.
* `batch_size_per_gpu` (int, default `64`): Batch size per GPU.
* `world_size` (int, default `8`): Number of distributed workers.
* `data_path` (str): Dataset root path.
* `best_structure_path` (str, optional): Path to `best_structure.txt` (absolute or relative to current working directory).



> Returns

* `Path` – Directory of retraining outputs.



#### Examples

```python
from ysautoml.network.zeroshot.mobilenetv2 import run_retrain_zeroshot

run_retrain_zeroshot(
    gpu_devices="0,1,2,3,4,5,6,7",
    epochs=150,
    init="custom_kaiming",
    best_structure_path="best_structure.txt"
)

```
