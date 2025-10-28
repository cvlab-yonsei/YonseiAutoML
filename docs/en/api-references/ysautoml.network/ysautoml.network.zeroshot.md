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
