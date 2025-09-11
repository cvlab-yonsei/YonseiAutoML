echo script name: $0

NUM_GPU=$1
SEED=$2

channel=8 #16
num_cells=8
dataset=cifar10
space=darts
BN=0

data_path=../data/cifar.python

save_dir=./SuperNet

CUDA_VISIBLE_DEVICES=${NUM_GPU} OMP_NUM_THREADS=4 python ./SuperNet/train_darts_ss.py \
	--data_path ${data_path} --dataset ${dataset} \
	--search_space_name ${space} \
	--channel ${channel} --layers ${num_cells} \
    --track_running_stats ${BN} \
	--config_path SuperNet/configs/OneShotFromAutoDL.config \
	--workers 4 --save_dir ${save_dir} \
	--print_freq 200 --rand_seed ${SEED} 
