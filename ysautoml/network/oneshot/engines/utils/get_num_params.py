import torch
from .get_strucs import get_struc

struc = get_struc()

def convert_arch(arch):
        arch = arch.tolist()
        arch = [arch[i].index(1) for i in range(len(arch))]
        return arch

def get_num_params(arch):
    # arch = struc[net_num]
    arch = convert_arch(arch)

    stem_params = 3 ** 2 * 3 * 16
    
    num_conv1s = arch.count(2)
    num_conv3s = arch.count(3)

    ch_square = ((16 ** 2) + (32 ** 2) + (64 ** 2)) * 5
    normal_cell_params = ch_square * 1 ** 2 * num_conv1s 
    normal_cell_params += ch_square * 3 ** 2 * num_conv3s 

    reduc_cell_params = 3 ** 2 * (16 * 32 + 32 **2) + 16 * 32 * 1**2 + 3 ** 2 * (32 * 64 + 64 **2) + 32 * 64 * 1**2
    
    fc_params = 64 * 10
    
    total_params = stem_params + normal_cell_params + reduc_cell_params + fc_params



    return total_params
