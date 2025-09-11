echo script name: $0

TAG=$1
SEED=$2

NUM_GPU=8

python ./Evaluation/train.py --tag ${TAG} --seed ${SEED} --num_gpus ${NUM_GPU} \
    --workers 4 --nesterov \
    --interval_ep_eval 20 \
    --arch 0 2 6 0 4 0 2 2 3 4 4 6 2 0 4 2 5 3 2 0 5
    #--max_epoch 300 --warmup \


#python ./Evaluation/train.py --tag ${TAG} --seed ${SEED} --num_gpus ${NUM_GPU} \
#    --workers 4 --nesterov \
#    --interval_ep_eval 20 \
#    --warmup \
#    --max_epoch 350 \
#    --search_space greedy \
#    --arch 6 4 12 6 2 4 6 8 6 4 8 10 6 0 0 4 4 6 9 0 11 
