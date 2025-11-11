<div align="center">
  <img src="https://github.com/user-attachments/assets/c9346208-f3d3-4977-bac2-3ba4036167e9" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">Yonsei CVLAB website</font></b>
    <sup>
      <a href="https://cvlab.yonsei.ac.kr/">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">CVLAB Open Sources</font></b>
    <sup>
      <a href="https://github.com/cvlab-yonsei">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>
  
[📘Documentation](https://dev-julie.gitbook.io/ysautoml) |
[🛠️Installation](https://dev-julie.gitbook.io/ysautoml/get-started/get-started#installation) |
[🚀Research](https://cvlab.yonsei.ac.kr/research/) |
[🤔Reporting Issues](https://github.com/cvlab-yonsei/YonseiAutoML/issues)

</div>

<div align="center">
<img src="https://github.com/user-attachments/assets/96ce99d6-c57f-41e8-9736-f20dc81369a0"/>
</div>

---

## Introduction

**YSAutoML** is an open-source library for the **automated construction and optimization of AI systems**, 
developed by Yonsei CVLab.  
It integrates key technologies across **data**, **network architectures**, and **optimization** 
to build task-specific, efficient deep learning solutions with minimal manual intervention.

The library is designed as part of a multi-year project that aims to:
- Develop a **next-generation automated AI platform** applicable to image recognition, segmentation, and object detection.
- Provide a unified solution that integrates dataset handling, model search, and efficient training.
- Support users in **rapidly generating customized AI models** according to their requirements (task, dataset, resource budget, etc.).

---

<details open>
<summary>✨ Major Features</summary>

- **Integrated Automation**  
  Combines dataset utilities, network architecture search (NAS), and optimization into a single pipeline.

- **Data Utilities**  
  Tools for dataset condensation, augmentation-aware BN (DSBN), and memory-efficient training.

- **Network Utilities (NAS)**  
  Support for few-shot, zero-shot, and one-shot NAS frameworks to automatically discover architectures.

- **Optimization Utilities**  
  Fixed-point quantization, loss function search, and multi-task loss integration for efficient deployment.

- **User-Centered Interface**  
  Accepts user requirements such as task type, dataset, computational budget, and memory constraints, 
  and generates optimized models automatically.

</details>

---

## What's New

💎 **Initial release of YSAutoML**: a unified library integrating dataset, network, and optimization utilities.

### Highlight

- Automated condensation of large datasets into compact synthetic sets.  
- DSBN (Dual BatchNorm) support for domain-adaptive training.  
- Unified NAS framework supporting multiple search paradigms.  
- Built-in support for quantization and multi-task loss training.

---

## Installation

YSAutoML requires **Python 3.8+** and **PyTorch 1.8+**.

```bash
git clone https://github.com/cvlab-yonsei/YonseiAutoML.git
cd YonseiAutoML
pip install -v -e .
```

---

## Getting Started

### Dataset Condensation with DSA
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

### Domain-Specific BatchNorm (DSBN)
```python
from ysautoml.data.dsbn import convert_and_wrap, train_with_dsbn

model = convert_and_wrap("resnet18_cifar", dataset="CIFAR10", num_classes=10)
result = train_with_dsbn(model, source_loader, target_loader, epochs=5, lr=0.01)
print(result["final_acc"])
```

---

## Module Overview

```
ysautoml
│
├── data
│   ├── fyi      # Dataset condensation (Flip Your Images)
│   ├── dsbn     # Domain-Specific BatchNorm
│
├── network
│   ├── fewshot  # Few-shot NAS
│   ├── zeroshot # Zero-shot NAS
│   ├── oneshot  # One-shot NAS
│
└── optimization
    ├── fxp        # Fixed-point quantization
    ├── losssearch # Loss search
    └── mtlloss    # Multi-task learning loss
```

---

## Roadmap

This project is part of a 4-year research initiative.  
In the final phase, we focus on **integrating and optimizing exploration spaces** for data, network, and loss jointly.

- **Unified Search Space Optimization**: jointly optimize network structures, data augmentation, and objectives.  
- **Automated Library Development**: provide a unified library that can generate AI models from user inputs.  
- **Efficiency Enhancements**: integrate dataset compression and fast optimization techniques to reduce memory and training cost.  

---

## FAQ

Please refer to the [documentation](https://dev-julie.gitbook.io/ysautoml/) for common usage questions.

---

## Acknowledgement

YSAutoML is developed by researchers at **Yonsei CVLab**.  
We thank all contributors and collaborators who helped design, implement, and test its components.

---

## Citation

If you use this toolbox or benchmark in your research, please cite:

```
@misc{ysautoml2025,
  title   = {YSAutoML: Automated AI System Construction Library},
  author  = {Yonsei CVLab},
  year    = {2025},
  note    = {https://github.com/cvlab-yonsei/YonseiAutoML}
}
```

---

## License

This project is released under the [Apache 2.0 license](LICENSE).
