import argparse

def make_parser():
    parser = argparse.ArgumentParser("Zero-Shot NAS (AutoFormer AZ-NAS)")
    parser.add_argument("--data_path", type=str, default="/dataset/ILSVRC2012")
    parser.add_argument("--population_num", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--param_limits", type=float, required=True)
    parser.add_argument("--min_param_limits", type=float, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--relative_position", action="store_true", default=True)
    parser.add_argument("--change_qkv", action="store_true", default=True)
    parser.add_argument("--dist_eval", action="store_true", default=True)
    parser.add_argument("--gp", action="store_true", default=True)
    return parser
