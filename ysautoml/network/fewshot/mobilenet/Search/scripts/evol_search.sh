echo script name: $0

SEED=$1

CKPT='baseline0-seed-0'

python ./Search/evol_search.py --seed ${SEED} --ckpt ${CKPT} \
    --workers 13 \
    --run_calib
