# Run DSA
ysautoml dsa --dataset CIFAR10 --model ConvNet --ipc 10 \
  --init real --lr_img 1.0 --num_exp 5 --num_eval 5 \
  --run_name DSAFYI --run_tags CIFAR10_10IPC --device 0 --eval_mode M

# Run DM
ysautoml dm --dataset CIFAR10 --model ConvNet --ipc 50 \
  --init real --Iteration 20000 --lr_img 1.0 \
  --run_name DMFYI --run_tags CIFAR10_50IPC --device 0 --eval_mode SS


# # resnet18_cifar를 DSBN으로 변환해서 state_dict 저장
# ysautoml dsbn \
#   --model resnet18_cifar \
#   --dataset CIFAR10 \
#   --num_classes 10 \
#   --use_aug \
#   --device 0 \
#   --export_path ./logs/dsbn_resnet18_cifar10.pth

ysautoml dsbn --model resnet18_cifar --dataset CIFAR10 --num_classes 10 \
  --epochs 10 --lr 0.1 --batch_size 128 --device 0 --mixed_batch

