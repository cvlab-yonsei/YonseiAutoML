echo script name: $0

NUM_GPU=$1
SEED_RUN=$2

P=50
G=20
save_dir=./Search

space=darts
dataset=cifar10
channel=16
num_cells=8
BN=0
data_path=../../data/cifar.python


seed=0
CKPT=seed-${seed}-dart-ss-baseline-8-16-64-250/seed-${seed}-last.pth


CUDA_VISIBLE_DEVICES=${NUM_GPU} OMP_NUM_THREADS=4 python ./Search/evol_search.py \
    --data_path ${data_path} --dataset ${dataset} \
    --search_space_name ${space} \
    --channel ${channel} --num_cells ${num_cells} \
    --track_running_stats ${BN} \
    --config_path SuperNet/configs/OneShotFromAutoDL.config \
    --workers 4 --save_dir ${save_dir} \
    --rand_seed ${SEED_RUN} \
    --ckpt ${CKPT} \
    --population_num ${P} \
    --max_epochs ${G} \
    --run_calib 0
#    --select_num ${K} \
#    --crossover_num ${CN} \
#    --mutation_num ${MN} \
#    --m_prob 0.1 \
