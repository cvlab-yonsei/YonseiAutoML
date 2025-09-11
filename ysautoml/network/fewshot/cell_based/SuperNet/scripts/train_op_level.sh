echo script name: $0

dataset=$1
BN=$2
NUM_GPU=$3
SEED=$4

channel=16
num_cells=5
max_nodes=4
space=nas-bench-201

data_path=../data/cifar.python
benchmark_file=../data/NAS-Bench-201-v1_0-e61699.pth

save_dir=./SuperNet

CUDA_VISIBLE_DEVICES=${NUM_GPU} OMP_NUM_THREADS=4 python ./SuperNet/train_op_level.py \
	--data_path ${data_path} --dataset ${dataset} \
	--search_space_name ${space} --max_nodes ${max_nodes} \
	--channel ${channel} --num_cells ${num_cells} \
    --track_running_stats ${BN} \
    --config_path SuperNet/configs/hi.config \
	--workers 4 --save_dir ${save_dir} --arch_nas_dataset ${benchmark_file} \
	--print_freq 200 --rand_seed ${SEED} --top_k 0.05 --option 5 
