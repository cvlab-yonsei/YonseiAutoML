from .parser import make_parser
from .engines.dsa_impl import run as _run_dsa_impl
from .engines.dm_impl import run as _run_dm_impl

def _build_args(**kwargs):
    parser = make_parser()
    # Initially parse with an empty list, then override the values with kwargs to match the CLI-style Namespace
    args = parser.parse_args([])
    for k, v in kwargs.items():
        if not hasattr(args, k):
            raise ValueError(f"Unknown arg: {k}")
        setattr(args, k, v)
    return args

def run_dsa(**kwargs):
    """
    Example:
      run_dsa(dataset="CIFAR10", model="ConvNet", ipc=10,
              dsa_strategy="color_crop_cutout_flip_scale_rotate",
              init="real", lr_img=1.0, num_exp=5, num_eval=5,
              run_name="DSAFYI", run_tags="CIFAR10_10IPC", device="0",
              eval_mode="M")
    """
    args = _build_args(method="DSA", **kwargs)
    return _run_dsa_impl(args)

def run_dm(**kwargs):
    """
    Example:
      run_dm(dataset="CIFAR10", model="ConvNet", ipc=10,
             dsa_strategy="color_crop_cutout_flip_scale_rotate",
             init="real", lr_img=1.0, num_exp=5, num_eval=5,
             run_name="DMFYI", run_tags="CIFAR10_10IPC", device="1",
             eval_mode="M")
    """
    args = _build_args(method="DC", **kwargs)
    return _run_dm_impl(args)
