echo script name: $0

SEED=0

NUM_GPU=8

#TAG=mobile0-k2rN-3638-TBS
#python ./SuperNet/train_decom.py --tag ${TAG} --seed ${SEED} --num_gpus ${NUM_GPU} \
#    --workers 4 --nesterov --num_K 2 --thresholds 36

TAG=mobile0-k4hN-TBS4
python ./SuperNet/train_decom.py --tag ${TAG} --seed ${SEED} --num_gpus ${NUM_GPU} \
    --workers 4 --nesterov --num_K 4 --thresholds 36 38 40

#TAG=mobile0-k6hN-TBS6
#python ./SuperNet/train_decom.py --tag ${TAG} --seed ${SEED} --num_gpus ${NUM_GPU} \
#    --workers 4 --nesterov --num_K 6 --thresholds 34 36 38 40 42
