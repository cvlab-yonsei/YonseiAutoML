echo script name: $0

dataset=$1
NUM_GPU=$2

channel=5 #16
num_cells=5
max_nodes=4
space=nas-bench-201

data_path="../../data/cifar.python"

save_dir=./Search
OUTPUT=./Search/logs
#CKPT=seed-5349-last.pth
CKPT=seed-0-opt5-wotrash-decom5-5-15-18-128-750/seed-0-last.pth
BN=0

CUDA_VISIBLE_DEVICES=${NUM_GPU} OMP_NUM_THREADS=4 python ./Search/search_320.py \
    --data_path ${data_path} --dataset ${dataset} \
    --search_space_name ${space} --max_nodes ${max_nodes} \
    --channel ${channel} --num_cells ${num_cells} \
    --track_running_stats ${BN} \
    --config_path SuperNet/configs/OneShotFromAutoDL.config \
    --workers 4 --save_dir ${save_dir} \
    --rand_seed 0 \
    --output_dir ${OUTPUT} \
    --ckpt ${CKPT}
