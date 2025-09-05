import typer
from .parser import make_parser
from .api import convert_and_wrap, train_with_dsbn_api

def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)

    # 1) 모델 준비
    model = convert_and_wrap(
        model_or_name=args.model,
        dataset=args.dataset,
        num_classes=args.num_classes,
        use_aug=args.use_aug,
        mode=args.mode,
        device=args.device,
        export_path=args.export_path,
    )

    # 2) 더미 예시 학습 (실제론 사용자 DataLoader 필요)
    # 여기서는 mixed_batch 여부에 따라 분기
    if args.mixed_batch:
        print("[DSBN] Running in mixed-batch mode (mode=3)")
        # train_with_dsbn_api(model, mixed_loader, epochs=args.epochs, lr=args.lr, mixed_batch=True)
    else:
        print("[DSBN] Running in separate source/target mode (mode=1/2)")
        # train_with_dsbn_api(model, source_loader, target_loader, epochs=args.epochs, lr=args.lr, mixed_batch=False)

    print("[DSBN] Finished")
