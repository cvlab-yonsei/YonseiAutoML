# ysautoml.data.dsbn

## funtional

### ysautoml.data.dsbn.convert\_and\_wrap

> ysautoml.data.dsbn.convert\_and\_wrap(\*\*kwargs)

Convert all `BatchNorm2d` layers in a model to **DSBN2d** (Domain-Specific BatchNorm) and set initial mode. DSBN maintains two sets of BN statistics (`BN_S` for source, `BN_T` for target/aug).

> Parameters

* **model\_or\_name** _(str or nn.Module)_:\
  Model name (factory-built, e.g. `"resnet18_cifar"`) or an existing model instance.
* **dataset** _(str, default `"CIFAR10"`)_: Dataset tag (reserved for model factory use).
* **num\_classes** _(int, default 10)_: Number of classes.
* **use\_aug** _(bool, default False)_: If `True` and `mode=None`, initial mode is set to 2 (Target BN). Otherwise 1 (Source BN).
* **mode** _(int or None, default None)_: DSBN mode.
  * `1`: use Source BN (`BN_S`)
  * `2`: use Target BN (`BN_T`)
  * `3`: split-half (first half BN\_S, second half BN\_T). Batch size must be even.\
    If `None`, inferred from `use_aug`.
* **device** _(str, default `"0"`)_: Device spec (sets `CUDA_VISIBLE_DEVICES` and moves model).
* **export\_path** _(str or None)_: If given, saves `model.state_dict()` to this path.



> Returns

* **nn.Module**: DSBN-converted model with mode set.



***

### ysautoml.data.dsbn.train\_with\_dsbn

> ysautoml.data.dsbn.train\_with\_dsbn(\*\*kwargs)

Train a DSBN-converted model using either **separate** or **mixed** batch training.

* **Separate mode** (`mixed_batch=False`):\
  Each iteration uses one source batch (mode=1) and one target batch (mode=2).
* **Mixed mode** (`mixed_batch=True`):\
  Each batch contains source+target samples concatenated. Model runs with mode=3, applying BN\_S to the first half and BN\_T to the second half.

> Parameters

* **model** _(nn.Module)_: DSBN-converted model.
* **train\_loader\_source** _(DataLoader)_: Source domain loader. In mixed mode, pass a mixed loader here.
* **train\_loader\_target** _(DataLoader, optional)_: Target loader (required for separate mode).
* **epochs** _(int, default 1)_: Number of training epochs.
* **lr** _(float, default 0.1)_: Learning rate.
* **mixed\_batch** _(bool, default False)_: If `True`, expects mixed batches and uses split-half mode.
* **device** _(str, default `"cuda"`)_: Training device.
* **log\_interval** _(int, default 10)_: Print loss every `log_interval` steps.



> Returns

* **dict** containing:
  * **logs** _(list)_: Training logs per step.
    * Separate mode: `(epoch, step, (loss_source, loss_target))`
    * Mixed mode: `(epoch, step, loss)`
  * **final\_acc** _(float)_: Final accuracy measured on source loader.
  * **state\_dict** _(OrderedDict)_: Trained model parameters.

***

#### Examples

```python

# separated batch training
from ysautoml.data.dsbn import convert_and_wrap, train_with_dsbn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# 1. preparing dataset (source/target split)
transform = transforms.Compose([
    transforms.ToTensor(),
])

full_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
len_source = len(full_train) // 2
len_target = len(full_train) - len_source
source_dataset, target_dataset = random_split(full_train, [len_source, len_target])

source_loader = DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=2)
target_loader = DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=2)

# 2. converting model (BN → DSBN)
model = convert_and_wrap("resnet18_cifar", dataset="CIFAR10", num_classes=10, use_aug=True, device="0")

# 3. training (seperated batch training mode)
result = train_with_dsbn(model, source_loader, target_loader,
                         epochs=2, lr=0.01, mixed_batch=True, device="cuda")

print("Final Accuracy:", result["final_acc"])
print("Number of log entries:", len(result["logs"]))

# if you want to save state_dict
import torch
torch.save(result["state_dict"], "./logs/dsbn_trained.pth")
```



***



<details>

<summary>Training log</summary>

```
Files already downloaded and verified
[Epoch 1/2][Step 0] LossS=2.4569 LossT=2.4533
[Epoch 1/2][Step 10] LossS=2.0590 LossT=2.1778
[Epoch 1/2][Step 20] LossS=1.7751 LossT=1.9871
[Epoch 1/2][Step 30] LossS=1.8834 LossT=1.9662
[Epoch 1/2][Step 40] LossS=1.5438 LossT=1.6438
[Epoch 1/2][Step 50] LossS=1.7033 LossT=1.5526
[Epoch 1/2][Step 60] LossS=1.8134 LossT=1.5944
[Epoch 1/2][Step 70] LossS=1.5852 LossT=1.5399
[Epoch 1/2][Step 80] LossS=1.3730 LossT=1.7473
[Epoch 1/2][Step 90] LossS=1.5869 LossT=1.4624
[Epoch 1/2][Step 100] LossS=1.5676 LossT=1.5956
[Epoch 1/2][Step 110] LossS=1.3916 LossT=1.2557
[Epoch 1/2][Step 120] LossS=1.5191 LossT=1.4593
[Epoch 1/2][Step 130] LossS=1.5009 LossT=1.5398
[Epoch 1/2][Step 140] LossS=1.4514 LossT=1.3022
[Epoch 1/2][Step 150] LossS=1.2466 LossT=1.4970
[Epoch 1/2][Step 160] LossS=1.4624 LossT=1.3039
[Epoch 1/2][Step 170] LossS=1.2874 LossT=1.2457
[Epoch 1/2][Step 180] LossS=1.2626 LossT=1.4146
[Epoch 1/2][Step 190] LossS=1.4181 LossT=1.3609
[Epoch 1/2] Done
[Epoch 2/2][Step 0] LossS=1.3235 LossT=1.3814
[Epoch 2/2][Step 10] LossS=1.1167 LossT=1.2604
[Epoch 2/2][Step 20] LossS=1.2077 LossT=1.2390
[Epoch 2/2][Step 30] LossS=1.2920 LossT=1.0327
[Epoch 2/2][Step 40] LossS=1.1390 LossT=1.0672
[Epoch 2/2][Step 50] LossS=1.1306 LossT=1.0909
[Epoch 2/2][Step 60] LossS=1.2758 LossT=1.1377
[Epoch 2/2][Step 70] LossS=1.2642 LossT=1.0378
[Epoch 2/2][Step 80] LossS=1.1297 LossT=1.1595
[Epoch 2/2][Step 90] LossS=1.0831 LossT=1.1323
[Epoch 2/2][Step 100] LossS=1.1231 LossT=1.2169
[Epoch 2/2][Step 110] LossS=1.3518 LossT=1.0639
[Epoch 2/2][Step 120] LossS=1.2794 LossT=1.3229
[Epoch 2/2][Step 130] LossS=0.9299 LossT=1.0228
[Epoch 2/2][Step 140] LossS=1.0112 LossT=1.2504
[Epoch 2/2][Step 150] LossS=1.1309 LossT=1.0886
[Epoch 2/2][Step 160] LossS=1.0641 LossT=1.0928
[Epoch 2/2][Step 170] LossS=1.1470 LossT=1.2876
[Epoch 2/2][Step 180] LossS=1.2428 LossT=1.2201
[Epoch 2/2][Step 190] LossS=1.0651 LossT=0.9680
[Epoch 2/2] Done
[Final Accuracy] 55.90%
Final Accuracy: 0.559
Number of log entries: 392
(yscvlab) root@d63c8bd87808:/data2/hyunju/data/TempAutoML# python api_test.py
Files already downloaded and verified
[Epoch 1/2][Step 0] Loss=2.5428
[Epoch 1/2][Step 10] Loss=2.0431
[Epoch 1/2][Step 20] Loss=2.1253
[Epoch 1/2][Step 30] Loss=1.7724
[Epoch 1/2][Step 40] Loss=1.7326
[Epoch 1/2][Step 50] Loss=1.8448
[Epoch 1/2][Step 60] Loss=1.8049
[Epoch 1/2][Step 70] Loss=1.7686
[Epoch 1/2][Step 80] Loss=1.6759
[Epoch 1/2][Step 90] Loss=1.4958
[Epoch 1/2][Step 100] Loss=1.5459
[Epoch 1/2][Step 110] Loss=1.5020
[Epoch 1/2][Step 120] Loss=1.4418
[Epoch 1/2][Step 130] Loss=1.4790
[Epoch 1/2][Step 140] Loss=1.5308
[Epoch 1/2][Step 150] Loss=1.4097
[Epoch 1/2][Step 160] Loss=1.6155
[Epoch 1/2][Step 170] Loss=1.6983
[Epoch 1/2][Step 180] Loss=1.4724
[Epoch 1/2][Step 190] Loss=1.2180
[Epoch 1/2] Done
[Epoch 2/2][Step 0] Loss=1.5312
[Epoch 2/2][Step 10] Loss=1.3657
[Epoch 2/2][Step 20] Loss=1.7006
[Epoch 2/2][Step 30] Loss=1.2808
[Epoch 2/2][Step 40] Loss=1.3028
[Epoch 2/2][Step 50] Loss=1.2232
[Epoch 2/2][Step 60] Loss=1.5063
[Epoch 2/2][Step 70] Loss=1.2261
[Epoch 2/2][Step 80] Loss=1.4344
[Epoch 2/2][Step 90] Loss=1.2264
[Epoch 2/2][Step 100] Loss=1.3558
[Epoch 2/2][Step 110] Loss=1.3469
[Epoch 2/2][Step 120] Loss=1.3965
[Epoch 2/2][Step 130] Loss=1.3787
[Epoch 2/2][Step 140] Loss=1.2738
[Epoch 2/2][Step 150] Loss=1.3671
[Epoch 2/2][Step 160] Loss=1.3588
[Epoch 2/2][Step 170] Loss=1.3039
[Epoch 2/2][Step 180] Loss=1.4443
[Epoch 2/2][Step 190] Loss=1.1007
[Epoch 2/2] Done
[Final Accuracy] 59.15%
Final Accuracy: 0.59148
Number of log entries: 392
```

</details>

