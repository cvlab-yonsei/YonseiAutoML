#!/bin/bash
#### 10000 archs
CUDA_VISIBLE_DEVICES=0, python3 search_autoformer_az.py --data-path '/dataset/ILSVRC2012' --gp --population-num 10000 --seed 123 \
  --param-limits 6 --min-param-limits 4 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/search/AZ-NAS/Tiny'

CUDA_VISIBLE_DEVICES=0, python3 search_autoformer_az.py --data-path '/dataset/ILSVRC2012' --gp --population-num 10000 --seed 123 \
 --param-limits 23 --min-param-limits 21 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-S.yaml' --output_dir './OUTPUT/search/AZ-NAS/Small'

 CUDA_VISIBLE_DEVICES=0, python3 search_autoformer_az.py --data-path '/dataset/ILSVRC2012' --gp --population-num 10000 --seed 123 \
 --param-limits 54 --min-param-limits 52 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-B.yaml' --output_dir './OUTPUT/search/AZ-NAS/Base'

#### 10000 archs // TFTAS budget
CUDA_VISIBLE_DEVICES=0, python3 search_autoformer_az.py --data-path '/dataset/ILSVRC2012' --gp --population-num 10000 --seed 123 \
  --param-limits 6.2 --min-param-limits 4 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/search/AZ-NAS-TFTAS_budget/Tiny'

CUDA_VISIBLE_DEVICES=0, python3 search_autoformer_az.py --data-path '/dataset/ILSVRC2012' --gp --population-num 10000 --seed 123 \
 --param-limits 24 --min-param-limits 21 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-S.yaml' --output_dir './OUTPUT/search/AZ-NAS-TFTAS_budget/Small'

 CUDA_VISIBLE_DEVICES=0, python3 search_autoformer_az.py --data-path '/dataset/ILSVRC2012' --gp --population-num 10000 --seed 123 \
 --param-limits 56.5 --min-param-limits 52 --change_qkv --relative_position --dist-eval --cfg './experiments/search_space/space-B.yaml' --output_dir './OUTPUT/search/AZ-NAS-TFTAS_budget/Base'