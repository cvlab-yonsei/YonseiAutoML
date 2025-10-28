## Implementation of AZ-NAS on the AutoFormer search space
- Prepare the ImageNet dataset in the directory `/dataset/ILSVRC2012`, or manually set the directory using the `--data-path` option in the shell script files.
- Refer the commands in the scripts files `search_autoformer_az.sh` and `train_searched_result_az.sh`.

## Credit
- The search space is proposed in [AutoFormer](https://github.com/microsoft/Cream/tree/b799630a29995163f282b15e2f38701160272fd1/AutoFormer)
- The code is modified from [TFTAS](https://github.com/decemberzhou/TF_TAS/tree/42616bcf1b6bb643bf968a8342f8aaddc4f53f32)

## Change Notes
- `./model/autoformer_subnet.py`: Modified from [`autoformer_space.py`](https://github.com/decemberzhou/TF_TAS/blob/42616bcf1b6bb643bf968a8342f8aaddc4f53f32/model/autoformer_space.py)
> * Generate a subnetwork for training
> * Add a function to extract blcok features

- `./model/space_engine.py`
> * Disable the Automatic Mixed Precision (AMP) option when a NaN loss error occurs

- `search_autoformer_az.py`: Modified from [`search_autoformer.py`](https://github.com/decemberzhou/TF_TAS/blob/42616bcf1b6bb643bf968a8342f8aaddc4f53f32/search_autoformer.py)
> * Search with zero-cost proxies of AZ-NAS
> * Add non-linear ranking aggregation

- `train_subnet.py`: Modified from [`train.py`](https://github.com/decemberzhou/TF_TAS/blob/42616bcf1b6bb643bf968a8342f8aaddc4f53f32/train.py)
> * Add tensorboard logging
> * Train with a subnet instead of a supernet