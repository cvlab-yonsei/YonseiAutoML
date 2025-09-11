echo script name: $0

dataset=$1
BN=$2
NUM_GPU=$3
SEED=$4

channel=5 #16
num_cells=5
max_nodes=4
space=nas-bench-201

data_path=../data/cifar.python

save_dir=./SuperNet

CUDA_VISIBLE_DEVICES=${NUM_GPU} OMP_NUM_THREADS=4 python ./SuperNet/train.py \
	--data_path ${data_path} --dataset ${dataset} \
	--search_space_name ${space} --max_nodes ${max_nodes} \
	--channel ${channel} --num_cells ${num_cells} \
    --track_running_stats ${BN} \
	--config_path SuperNet/configs/OneShotFromAutoDL.config \
	--workers 4 --save_dir ${save_dir} \
	--print_freq 200 --rand_seed ${SEED} 
