from ysautoml.data.fyi import run_dsa, run_dm
# from ysautoml.optimization import fxp_quantize
# from ysautoml.network import run_fewshot_nas

# Dataset condensation
run_dsa(dataset="CIFAR10", model="ConvNet", ipc=10, device="0")

# # Quantization
# fxp_quantize(model, dataset, w_bits=4, a_bits=8, g_bits=8)

# # Few-shot NAS
# search_result = run_fewshot_nas(search_space="vit.yaml", dataset="CIFAR100")


# from ysautoml import run_dsa, run_dm

# run_dsa(
#     dataset="CIFAR10",
#     model="ConvNet",
#     ipc=10,
#     dsa_strategy="color_crop_cutout_flip_scale_rotate",
#     init="real", lr_img=1.0, num_exp=5, num_eval=5,
#     run_name="DSAFYI", run_tags="CIFAR10_10IPC", device="0", eval_mode="M",
# )

# run_dm(
#     dataset="CIFAR10",
#     model="ConvNet",
#     ipc=10,
#     dsa_strategy="color_crop_cutout_flip_scale_rotate",
#     init="real", lr_img=1.0, num_exp=5, num_eval=5,
#     run_name="DMFYI", run_tags="CIFAR10_10IPC", device="1", eval_mode="M",
# )
