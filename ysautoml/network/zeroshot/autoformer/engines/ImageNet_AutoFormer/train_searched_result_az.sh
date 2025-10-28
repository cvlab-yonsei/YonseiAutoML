#!/bin/bash
#### 256x8 // 512x4

# Tiny / plain budget
CUDA_LAUNCH_BLOCKING=1 \
python3 -m torch.distributed.launch --master_port 6666 --nproc_per_node=8 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --epochs 500 --warmup-epochs 20 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/AZ-NAS/Tiny.yaml' --output_dir './OUTPUT/AZ-NAS/Tiny-bs256x8-use_subnet-500ep'

# Small / plain budget
CUDA_LAUNCH_BLOCKING=1 \
python3 -m torch.distributed.launch --master_port 6666 --nproc_per_node=8 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --epochs 500 --warmup-epochs 20 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/AZ-NAS/Small.yaml' --output_dir './OUTPUT/AZ-NAS/Small-bs256x8-use_subnet-500ep'

# Base / plain budget / 300ep
CUDA_LAUNCH_BLOCKING=1 \
python3 -m torch.distributed.launch --master_port 6666 --nproc_per_node=8 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --epochs 300 --warmup-epochs 20 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/AZ-NAS/Base.yaml' --output_dir './OUTPUT/AZ-NAS/Base-bs256x8-use_subnet-300ep'

 # Tiny / TFTAS budget
CUDA_LAUNCH_BLOCKING=1 \
python3 -m torch.distributed.launch --master_port 6666 --nproc_per_node=8 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --epochs 500 --warmup-epochs 20 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/AZ-NAS-TFTAS_budget/Tiny.yaml' --output_dir './OUTPUT/AZ-NAS-TFTAS_budget/Tiny-bs256x8-use_subnet-500ep'

 # Small / TFTAS budget
CUDA_LAUNCH_BLOCKING=1 \
python3 -m torch.distributed.launch --master_port 6666 --nproc_per_node=8 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --epochs 500 --warmup-epochs 20 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/AZ-NAS-TFTAS_budget/Small.yaml' --output_dir './OUTPUT/AZ-NAS-TFTAS_budget/Small-bs256x8-use_subnet-500ep'

# Base / TFTAS budget / 300ep
CUDA_LAUNCH_BLOCKING=1 \
python3 -m torch.distributed.launch --master_port 7777 --nproc_per_node=8 --use_env train_subnet.py --data-path '/dataset/ILSVRC2012' --gp --change_qkv --relative_position \
 --epochs 300 --warmup-epochs 20 --batch-size 256 --mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/AZ-NAS-TFTAS_budget/Base.yaml' --output_dir './OUTPUT/AZ-NAS-TFTAS_budget/Base-bs256x8-use_subnet-300ep'