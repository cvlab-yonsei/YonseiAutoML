mkdir -p logs

GPU=0
seed=0 


weighting=GeMTL
# Arithmetic, GLS, UW, DWA, RLW, GradNorm, SI, IMTL_L, LSBwD, LSBwoD, AMTL, GeMTL

arch=HPS

python main.py \
--weighting ${weighting} \
--arch ${arch} \
--dataset_path /dataset/nyuv2 \
--gpu_id ${GPU} \
--seed ${seed} \
--scheduler step \
--mode train 

#>> logs/${arch}_${weighting}_seed${seed}.txt