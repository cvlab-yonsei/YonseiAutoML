# GET STARTED

## Prerequisites

In this section, we demonstrate how to prepare an environment with PyTorch.

YSAutoML works on Linux, Windows, and macOS. It requires Python 3.8, CUDA 11,1, and PyTorch 1.9.1.

### Method 1: Using the Official Docker Container (Recommended for GPU)

For guaranteed environment consistency and to handle complex CUDA dependencies (`torchvision==0.10.1+cu111`), we strongly recommend using the official Docker setup.

**Step 0.** Clone the Repository.

The required `Dockerfile` is located in the root of the GitHub repository.

```shellscript
git clone [https://github.com/cvlab-yonsei/YonseiAutoML.git](https://github.com/cvlab-yonsei/YonseiAutoML.git)
cd YonseiAutoML
```

**Step 1.** Build the Docker Image.

Use the cloned `Dockerfile` to build your environment.

```shellscript
docker build -t ysautoml:0.1.1 .
```

**Step 2.** Run the Container.

Run the container, exposing necessary resources (e.g., GPU support).

```shellscript
docker run -it --gpus all ysautoml:0.1.1 /bin/bash
```

## Installation

Once inside the container, execute the comprehensive `pip install` command, which includes all necessary index URLs (TestPyPI, official PyPI, and PyTorch CUDA channel).

```shellscript
pip install \
    --index-url [https://test.pypi.org/simple/](https://test.pypi.org/simple/) \
    --extra-index-url [https://pypi.org/simple/](https://pypi.org/simple/) \
    --extra-index-url [https://download.pytorch.org/whl/cu111](https://download.pytorch.org/whl/cu111) \
    ysautoml==0.1.1
```

### Install on Google Colab

[Google Colab](https://colab.research.google.com/) usually has PyTorch installed, thus we only need to install YSAutoML with the following commands.

**Step 1.** Install YSAutoML.

```shellscript
!git clone https://github.com/cvlab-yonsei/YonseiAutoML.git
%cd YonseiAutoML
!pip install -e .
```

**Step 2.** Verification

```shellscript
import ysautoml
print(ysautoml.__version__)
# Example output: 0.1.1, or an another version.
```

{% hint style="info" %}
Within Jupyter, the exclamation mark `!` is used to call external executables and `%cd` is a [magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd) to change the current working directory of Python.
{% endhint %}

## Troubleshooting

If you have some issues during the installation, please first view the FAQ page. You may [open an issue](https://github.com/cvlab-yonsei/YonseiAutoML/issues) on GitHub if no solution is found.
