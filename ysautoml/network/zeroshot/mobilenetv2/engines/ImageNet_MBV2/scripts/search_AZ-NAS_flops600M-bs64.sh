#!/bin/bash
cd "$(dirname "$0")"
set -e

# echo -e "Enter the metric"
# read metric
# echo -e "Enter the gpu"
# read gpu
# echo -e "Enter the seed"
# read seed
# echo "Run this script with metric=$metric, search gpu=$gpu, seed=$seed"

# if [ "$#" -lt 2 ] || [ "$#" -gt 2 ]; then
#     echo "$# is Illegal number of parameters."
#     echo "Usage: *.sh search_gpu seed"
# 	exit 1
# fi

metric=AZ_NAS
population_size=1024
evolution_max_iter=1e5

gpu=0
seed=123
echo "Run this script with search_gpu=$gpu, seed=$seed"

cd ../
save_dir=./save_dir/${metric}_flops600M-searchbs64-pop${population_size}-iter${evolution_max_iter}-${seed}
mkdir -p ${save_dir}
evolution_max_iter=$(printf "%.0f" $evolution_max_iter)

resolution=224
budget_flops=600e6
max_layers=14
epochs=150

echo "SuperConvK3BNRELU(3,8,2,1)SuperResIDWE6K3(8,32,2,8,1)SuperResIDWE6K3(32,48,2,32,1)\
SuperResIDWE6K3(48,96,2,48,1)SuperResIDWE6K3(96,128,2,96,1)\
SuperConvK1BNRELU(128,2048,1,1)" > ${save_dir}/init_plainnet.txt

python evolution_search_az.py --gpu ${gpu} \
  --zero_shot_score ${metric} \
  --search_space SearchSpace/search_space_IDW_fixfc.py \
  --budget_flops ${budget_flops} \
  --max_layers ${max_layers} \
  --batch_size 64 \
  --input_image_size ${resolution} \
  --plainnet_struct_txt ${save_dir}/init_plainnet.txt \
  --num_classes 1000 \
  --evolution_max_iter ${evolution_max_iter} \
  --population_size ${population_size} \
  --save_dir ${save_dir} \
  --dataset imagenet-1k \
  --num_worker 0 \
  --rand_input True \
  --search_no_res False \
  --seed ${seed} \
  --datapath /dataset/ILSVRC2012/

python analyze_model.py \
  --input_image_size 224 \
  --num_classes 1000 \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt