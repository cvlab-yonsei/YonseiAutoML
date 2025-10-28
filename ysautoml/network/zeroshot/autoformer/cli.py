from .parser import make_parser
from .api import run_search_zeroshot

def main():
    parser = make_parser()
    args = parser.parse_args()

    run_search_zeroshot(
        data_path=args.data_path,
        population_num=args.population_num,
        seed=args.seed,
        param_limits=args.param_limits,
        min_param_limits=args.min_param_limits,
        cfg=args.cfg,
        output_dir=args.output_dir,
        device=args.device,
        relative_position=args.relative_position,
        change_qkv=args.change_qkv,
        dist_eval=args.dist_eval,
        gp=args.gp
    )

if __name__ == "__main__":
    main()
