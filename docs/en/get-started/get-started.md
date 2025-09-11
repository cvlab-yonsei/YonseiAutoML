# GET STARTED

## Prerequisites

In this section, we demonstrate how to prepare an environment with PyTorch.

YSAutoML works on Linux, Windows, and macOS. It requires Python 3.7+, CUDA 9.2+, and PyTorch 1.8+.

{% hint style="info" %}
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation). Otherwise, you can follow these steps for the preparation.
{% endhint %}

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```
conda create --name yscvlab python=3.8 -y
conda activate yscvlab
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```
conda install pytorch torchvision cpuonly -c pytorch
```



### Use YSAutoML with virtualenv

**Step 0.** Install virtualenv with python 3.7+.

```
apt install python3.8-venv
```

**Step 1.** Create a virtual environment and activate it.

<pre><code><strong>python3 -m venv yscvlab
</strong>source yscvlab/bin/activate
</code></pre>

**Step 2.** Install PyTorch.

```
pip install torch torchvision
```





## Installation

**Step 1.** Install YSAutoML.

```bash
git clone https://github.com/cvlab-yonsei/TempAutoML.git
cd TempAutoML
pip install -v -e
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```



### Install on Google Colab

[Google Colab](https://colab.research.google.com/) usually has PyTorch installed, thus we only need to install YSAutoML with the following commands.

**Step 1.** Install YSAutoML.

```
!git clone https://github.com/cvlab-yonsei/TempAutoML.git
%cd TempAutoML
!pip install -e .
```

**Step 2.** Verification

```
import ysautoml
print(ysautoml.__version__)
# Example output: 3.0.0, or an another version.
```

{% hint style="info" %}
Within Jupyter, the exclamation mark `!` is used to call external executables and `%cd` is a [magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd) to change the current working directory of Python.
{% endhint %}



## Troubleshooting

If you have some issues during the installation, please first view the FAQ page. You may [open an issue](https://github.com/cvlab-yonsei/TempAutoML/issues) on GitHub if no solution is found.



