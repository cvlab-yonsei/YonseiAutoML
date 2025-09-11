echo script name: $0

GPU=$1
SEED=$2

save_dir=./Evaluation
arch='|dil_sepc_3x3~0|avg_pool_3x3~1|+|dua_sepc_5x5~0|skip_connect~1|dua_sepc_3x3~2|+|dua_sepc_3x3~0|dua_sepc_5x5~1|dua_sepc_5x5~2|none~3|+|dua_sepc_3x3~0|dua_sepc_5x5~1|dil_sepc_5x5~2|dua_sepc_5x5~3|none~4|'

python ./Evaluation/train.py --arch ${arch} \
                             --rand_seed ${SEED} --save_dir ${save_dir} \
                             --gpu ${GPU} \
                             --auxiliary --cutout
