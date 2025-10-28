## Implementation of AZ-NAS on the MobileNetV2 search space
- Prepare the ImageNet dataset in the directory `/dataset/ILSVRC2012`, or manually change the directory specified in `./Dataloader/__init__.py` (Lines 33-34).
- Run the scripts in the `scripts` folder. For example:
```bash
cd scripts

# find a network architecture with a FLOPs constraint of 450M
./search_AZ-NAS_flops450M-bs64.sh 

# train a selected network for 480 epochs with the teacher-student distillation and advanced data augmentation techniques
./train_AZ-NAS_flops450M-480ep-bs64x8.sh 
# or train a selected network for 150 epochs with a simplified training setting
./train_AZ-NAS_flops450M-150ep-bs64x8.sh 
```

## Credit
- The code is modified from [ZenNAS](https://github.com/idstcv/ZenNAS/tree/d1d617e0352733d39890fb64ea758f9c85b28c1a) and [ZiCo](https://github.com/SLDGroup/ZiCo/tree/b0fec65923a90e84501593f675b1e2f422d79e3d)

## Change Notes
- `Masternet.py`
> *  Add a function to extract block features

- `evolutionary_search_az.py`: Modified from [`evolutionary_search.py`](https://github.com/SLDGroup/ZiCo/blob/b0fec65923a90e84501593f675b1e2f422d79e3d/evolution_search.py)
> *  Implement an evolutionary search algorithm for AZ-NAS

- `train_image_classification.py` and `ts_train_image_classification.py`
> *  Add Kaiming normal initialization