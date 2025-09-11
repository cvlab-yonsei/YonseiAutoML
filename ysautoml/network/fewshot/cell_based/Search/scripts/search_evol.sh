echo script name: $0

NUM_GPU=$1
SEED_RUN=$2

P=50
G=20
save_dir=./Search

space=nas-bench-201
dataset=cifar10
channel=5 #16
num_cells=5
max_nodes=4
BN=0
data_path=../../data/cifar.python
benchmark_file=../../data/NAS-Bench-201-v1_0-e61699.pth


seed=0
#CKPT=seed-${seed}-opt5-wotrash-baseline-16-128-1000/seed-${seed}-last.pth
CKPT=seed-${seed}-opt5-wotrash-decom5-5-15-20-128-1000-balanced/seed-${seed}-last.pth # K=3


CUDA_VISIBLE_DEVICES=${NUM_GPU} OMP_NUM_THREADS=4 python ./Search/evol_search.py \
    --data_path ${data_path} --dataset ${dataset} \
    --search_space_name ${space} --max_nodes ${max_nodes} \
    --channel ${channel} --num_cells ${num_cells} \
    --track_running_stats ${BN} \
    --config_path SuperNet/configs/OneShotFromAutoDL.config \
    --workers 4 --save_dir ${save_dir} --arch_nas_dataset ${benchmark_file} \
    --rand_seed ${SEED_RUN} \
    --ckpt ${CKPT} \
    --population_num ${P} \
    --max_epochs ${G} \
    --run_calib 0
#    --select_num ${K} \
#    --crossover_num ${CN} \
#    --mutation_num ${MN} \
#    --m_prob 0.1 \
