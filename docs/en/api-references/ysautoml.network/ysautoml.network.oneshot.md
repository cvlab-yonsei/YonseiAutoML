# ysautoml.network.oneshot

## Funtional Modules

* `ysautoml.network.oneshot.train_dynas`
* `ysautoml.network.oneshot.quantize_dynas`

#### ysautoml.network.oneshot.train\_dynas

> ysautoml.network.oneshot.train\_dynas(\*\***kwargs)**

Run **DYNAS** (Subnet-Aware Dynamic Supernet Training for Neural Architecture Search) or **SPOS baseline** training using the original `train_spos.py` script, integrated into the YSAutoML API system.

> Parameters

* `log_dir` (`str`, default `logs/spos_dynamic`): Directory path to store TensorBoard and log outputs.
* `file_name` (`str`, default `spos_dynamic`): Base name for experiment log and validation pickle files.
* `seed` (`int`, default `0`): Random seed for reproducibility.
* `epochs` (`int`, default `250`): Total number of training epochs.
* `lr` (`float`, default `0.025`): Initial learning rate for SGD optimizer.
* `momentum` (`float`, default `0.9`): Momentum factor for SGD optimization.
* `wd` (`float`, default `0.0005`): Weight decay coefficient for L2 regularization.
* `nesterov` (`bool`, default `True`): Whether to enable Nesterov momentum in SGD.
* `train_batch_size` (`int`, default `64`): Batch size used for training dataset.
* `val_batch_size` (`int`, default `256`): Batch size used for validation dataset.
* `method` (`str`, default `dynas`): Training mode. Use `baseline` for SPOS baseline, or `dynas` for dynamic architecture training with adaptive learning rate scheduling.
* `max_coeff` (`float`, default `4.0`): Maximum coefficient `γ_max` controlling adaptive learning rate scaling for DYNAS.



> Returns

* `None` :  All training logs, pickled validation results, and TensorBoard summaries are saved under `log_dir`.
  * Logs: `${log_dir}/spos_dynamic.log`
  * TensorBoard events: `${log_dir}/events.out.tfevents.*`
  * Validation accuracy pickle: `exps/NAS-Bench-201-algos/valid_accs/{file_name}.pkl`
  * Kendall correlation report (CIFAR10/100, ImageNet)





#### Examples

```python
from ysautoml.network.oneshot import train_dynas

# Run DYNAS (Dynamic NAS)
train_dynas(
    log_dir="./logs/dynas_exp1",
    file_name="dynas_c10",
    seed=42,
    epochs=5,
    method="dynas",          # DYNAS adaptive LR mode
    max_coeff=4.0
)

# Run SPOS Baseline
train_dynas(
    log_dir="./logs/spos_exp1",
    file_name="spos_baseline",
    seed=0,
    epochs=5,
    method="baseline"        # vanilla SPOS
)

```

***

#### ysautoml.network.oneshot.quantize\_dynas

> ysautoml.network.oneshot.quantize\_dynas(\*\***kwargs)**

Run **post-training quantization** for a searched DYNAS/SPOS subnet using the original comparison.py script, integrated into the YSAutoML API system. This function loads the searched architecture from a text file, restores a trained checkpoint, applies uniform k-bit quantization, and saves the quantized checkpoint automatically.

> Parameters

* `arch_file` (`str`, default `best_structure.txt`): Path to the text file containing the searched architecture configuration. The file must contain comma-separated integers (e.g., `2, 2, 6, 0, ...`).
* `ckpt` (`str`, default `epoch-241.pt`): Path to the trained floating-point checkpoint file to be quantized.
* `quant_bits` (`int`, default `8`): Bit-width for uniform weight quantization (e.g., 8-bit, 6-bit).
* `log_name` (`int`, default `quantize_output`): Tag name for quantization process logging.



> Returns

*   `None` : &#x20;

    All quantization results are saved automatically as a new checkpoint file.

    * Quantized checkpoint:\
      `${ckpt_dir}/${ckpt_name}_quant{quant_bits}.pt`
    * Console output:
      * Loaded architecture tuple
      * Quantization status
      * Final saved checkpoint path

#### Examples

```python
from ysautoml.network.oneshot import quantize_dynas

# Run 8-bit post-training quantization
quantize_dynas(
    arch_file="best_architecture.txt",
    ckpt="epoch-241.pt",
    quant_bits=8
)

# Run lower-bit quantization (e.g., 6-bit)
quantize_dynas(
    arch_file="./outputs/best_architecture.txt",
    ckpt="./checkpoints/epoch-250.pt",
    quant_bits=6,
    log_name="quant6_test"
)
```

