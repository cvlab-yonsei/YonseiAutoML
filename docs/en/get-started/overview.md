---
description: >-
  This chapter introduces you to the framework of YSAutoML, and provides links
  to detailed tutorials about YSAutoML
---

# OVERVIEW

## What is YSAutoML

<figure><img src="../.gitbook/assets/library_structure.png" alt=""><figcaption></figcaption></figure>

**YSAutoML** is an open-source library designed for the **automated construction of AI systems**.\
It integrates independent utilities across **data, network, and optimization** into a unified platform, enabling efficient model building for tasks such as **image recognition, segmentation, and object detection**.

The project is developed by **Yonsei CVLab**, as part of a multi-year initiative on building a **next-generation automated AI platform**.

## How to Use this Guide

Here is a detailed step-by-step guide to learn more about YSAutoML:

1. For installation instructions, please see get\_started.



## Major Features

* **Data Utilities**
  * **fyi**: Dataset condensation (Flip Your Images), reducing large datasets into compact synthetic sets.
  * **dsbn**: Domain-Specific BatchNorm for source/target or augmentation-aware training.
* **Network Utilities (NAS)**
  * **fewshot**: Few-shot NAS, exploring architectures with limited data.
  * **zeroshot**: Zero-shot NAS, searching architectures without labeled training data.
  * **oneshot**: One-shot NAS, training a supernet and searching subnets efficiently.
* **Optimization Utilities**
  * **fxp**: Fixed-point quantization (W/A/G bitwidth 4–8).
  * **losssearch**: Automated loss function search.
  * **mtlloss**: Multi-task loss integration for classification, segmentation, and beyond.
* **Unified Workflow**
  * Users specify task, dataset, resource budget, and memory constraints.
  * YSAutoML automatically builds and trains optimized AI systems under these requirements.





## Library Structure

```nginx
ysautoml
│
├── data
│   ├── fyi      # Dataset condensation
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
    └── mtlloss    # Multi-task loss

```



## License

This project is released under the Apache 2.0 License\


