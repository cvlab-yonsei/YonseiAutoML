echo script name: $0

dataset=$1
NUM_GPU=$2
SPLIT=$3
SEED=$4

channel=8 #5 #16
num_cells=5
max_nodes=4
space=nas-bench-201

data_path="../../data/cifar.python"

save_dir=./Search
OUTPUT=./Search/logs
#SEED=0

#CKPT=seed-${SEED}-opt5-wotrash/seed-${SEED}-last.pth
#CKPT=seed-${SEED}-opt5-wotrash-baseline-5-128-1000-balanced/seed-${SEED}-last.pth
CKPT=seed-${SEED}-opt5-wotrash-decom5-8-25-64-500-balanced/seed-${SEED}-last.pth # K=2
#CKPT=seed-${SEED}-opt5-wotrash-decom5-5-10-30-64-750-balanced/seed-${SEED}-last.pth # K=3
#CKPT=seed-${SEED}-opt5-wotrash-decom5-4-10-20-30-128-1000/seed-${SEED}-last.pth # K=4
BN=0

CUDA_VISIBLE_DEVICES=${NUM_GPU} OMP_NUM_THREADS=4 python ./Search/search_all.py \
    --data_path ${data_path} --dataset ${dataset} \
    --search_space_name ${space} --max_nodes ${max_nodes} \
    --channel ${channel} --num_cells ${num_cells} \
    --track_running_stats ${BN} \
    --config_path SuperNet/configs/OneShotFromAutoDL.config \
    --workers 4 --save_dir ${save_dir} \
    --rand_seed 0 \
    --output_dir ${OUTPUT} --edge_op ${SPLIT} \
    --ckpt ${CKPT}
