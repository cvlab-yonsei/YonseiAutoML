# from ysautoml.network.zeroshot.autoformer import run_search_zeroshot

# run_search_zeroshot(
#     param_limits=6,
#     min_param_limits=4,
#     cfg="space-T.yaml",   # 이름만 주면
#     output_dir="./OUTPUT/search/AZ-NAS/Tiny"
# )

# ##################

# from ysautoml.network.zeroshot.autoformer import run_retrain_zeroshot

# run_retrain_zeroshot(
#     cfg="./Tiny.yaml",
#     output_dir="./OUTPUT/AZ-NAS/Tiny-bs256x8-use_subnet-500ep",
#     epochs=500
# )

# ########################

# from ysautoml.network.zeroshot.mobilenetv2 import run_search_zeroshot

# # 기본 실행
# run_search_zeroshot(
#     gpu=0,
#     seed=123,
#     metric="AZ_NAS",
#     # population_size=1024,
#     # evolution_max_iter=int(1e5),
#     population_size=100,
#     evolution_max_iter=100,
#     resolution=224,
#     budget_flops=1e9,
#     max_layers=16,
#     # batch_size=64,
#     batch_size=32,
#     data_path="/dataset/ILSVRC2012/",
# )

# # Small (450M)
# run_search_zeroshot(
#     gpu=0,
#     seed=123,
#     budget_flops=450e6,
#     max_layers=14,
# )

# # Medium (600M)
# run_search_zeroshot(
#     gpu=0,
#     seed=123,
#     budget_flops=600e6,
#     max_layers=14,
# )

# # Large (1G)
# run_search_zeroshot(
#     gpu=0,
#     seed=123,
#     budget_flops=1000e6,
#     max_layers=16,
# )

# #################

# from ysautoml.network.zeroshot.mobilenetv2 import run_retrain_zeroshot

# # retrain 실행
# run_retrain_zeroshot(
#     # gpu_devices="0,1,2,3,4,5,6,7",
#     gpu_devices="0,1,2,3",
#     world_size=4,
#     # epochs=150,
#     epochs=1,
#     warmup=0,
#     init="custom_kaiming",
#     # best_structure_path="best_structure.txt"
#     best_structure_path="./ysautoml/network/zeroshot/mobilenetv2/engines/ImageNet_MBV2/save_dir/AZ_NAS_flops1G-searchbs32-pop100-iter100-123/best_structure.txt"
# )

##########


# from ysautoml.network.oneshot import train_dynas

# train_dynas(
#     log_dir="logs/spos_dynamic",
#     file_name="spos_dynamic",
#     method="dynas",
#     save_path="./results"
# )

from ysautoml.network.oneshot import train_dynas

train_dynas(
    log_dir="./logs/dynas_test",
    file_name="spos_dynamic",
    seed=42,
    epochs=5,
    method="dynas"
)
