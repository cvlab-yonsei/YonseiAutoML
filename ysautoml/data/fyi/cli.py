import sys
from .parser import make_parser
from .engines.dsa_impl import run as _run_dsa_impl
from .engines.dm_impl import run as _run_dm_impl

def main_dsa(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    args.method = "DSA"
    return _run_dsa_impl(args)

def main_dm(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    args.method = "DC"
    return _run_dm_impl(args)

if __name__ == "__main__":
    # for debugging
    sys.exit(main_dsa())
