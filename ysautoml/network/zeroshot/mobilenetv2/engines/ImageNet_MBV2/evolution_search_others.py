'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''


import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse, random, logging, time
import torch
from torch import nn
import numpy as np
import global_utils
import Masternet
import PlainNet
# from tqdm import tqdm
from xautodl import datasets
import time

from ZeroShotProxy import compute_zen_score, compute_te_nas_score, compute_syncflow_score, compute_gradnorm_score, compute_NASWOT_score, compute_zico
import benchmark_network_latency

working_dir = os.path.dirname(os.path.abspath(__file__))

def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--zero_shot_score', type=str, default=None,
                        help='could be: ZiCo (form ZiCo); params, (for \#Params); Zen (for Zen-NAS); TE (for TE-NAS)')
    parser.add_argument('--search_space', type=str, default=None,
                        help='.py file to specify the search space.')
    parser.add_argument('--evolution_max_iter', type=int, default=int(100000),
                        help='max iterations of evolution.')
    parser.add_argument('--budget_model_size', type=float, default=None, help='budget of model size ( number of parameters), e.g., 1e6 means 1M params')
    parser.add_argument('--budget_flops', type=float, default=None, help='budget of flops, e.g. , 1.8e6 means 1.8 GFLOPS')
    parser.add_argument('--budget_latency', type=float, default=None, help='latency of forward inference per mini-batch, e.g., 1e-3 means 1ms.')
    parser.add_argument('--max_layers', type=int, default=None, help='max number of layers of the network.')
    parser.add_argument('--batch_size', type=int, default=None, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=None,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--population_size', type=int, default=512, help='population size of evolution.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='output directory')
    parser.add_argument('--gamma', type=float, default=1e-2,
                        help='noise perturbation coefficient')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='number of classes')
    parser.add_argument('--dataset', type=str,
                        help='type of dataset')
    parser.add_argument('--datapath', type=str,
                        help='root of path')
    parser.add_argument('--num_worker', type=int, default=40,
                        help='root of path')
    parser.add_argument('--maxbatch', type=int, default=2,
                        help='root of path')
                        
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt

def get_new_random_structure_str(AnyPlainNet, structure_str, num_classes, get_search_space_func,
                                 num_replaces=1):
    the_net = AnyPlainNet(num_classes=num_classes, plainnet_struct=structure_str, no_create=True)
    assert isinstance(the_net, PlainNet.PlainNet)
    selected_random_id_set = set()
    for replace_count in range(num_replaces):
        random_id = random.randint(0, len(the_net.block_list) - 1)
        if random_id in selected_random_id_set:
            continue
        selected_random_id_set.add(random_id)
        to_search_student_blocks_list_list = get_search_space_func(the_net.block_list, random_id)

        to_search_student_blocks_list = [x for sublist in to_search_student_blocks_list_list for x in sublist]
        new_student_block_str = random.choice(to_search_student_blocks_list)

        if len(new_student_block_str) > 0:
            new_student_block = PlainNet.create_netblock_list_from_str(new_student_block_str, no_create=True)
            assert len(new_student_block) == 1
            new_student_block = new_student_block[0]
            if random_id > 0:
                last_block_out_channels = the_net.block_list[random_id - 1].out_channels
                new_student_block.set_in_channels(last_block_out_channels)
            the_net.block_list[random_id] = new_student_block
        else:
            # replace with empty block
            the_net.block_list[random_id] = None
    pass  # end for

    # adjust channels and remove empty layer
    tmp_new_block_list = [x for x in the_net.block_list if x is not None]
    last_channels = the_net.block_list[0].out_channels
    for block in tmp_new_block_list[1:]:
        block.set_in_channels(last_channels)
        last_channels = block.out_channels
    the_net.block_list = tmp_new_block_list

    new_random_structure_str = the_net.split(split_layer_threshold=6)
    return new_random_structure_str


def get_splitted_structure_str(AnyPlainNet, structure_str, num_classes):
    the_net = AnyPlainNet(num_classes=num_classes, plainnet_struct=structure_str, no_create=True)
    assert hasattr(the_net, 'split')
    splitted_net_str = the_net.split(split_layer_threshold=6)
    return splitted_net_str

def get_latency(AnyPlainNet, random_structure_str, gpu, args):
    the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                            no_create=False, no_reslink=False)
    if gpu is not None:
        the_model = the_model.cuda(gpu)
    the_latency = benchmark_network_latency.get_model_latency(model=the_model, batch_size=args.batch_size,
                                                              resolution=args.input_image_size,
                                                              in_channels=3, gpu=gpu, repeat_times=1,
                                                              fp16=True)
    del the_model
    torch.cuda.empty_cache()
    return the_latency

def compute_nas_score(AnyPlainNet, random_structure_str, gpu, args, trainloader=None, lossfunc=None):
    # compute network zero-shot proxy score
    the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                            no_create=False, no_reslink=True)
    the_model = the_model.cuda(gpu)
    
    if args.zero_shot_score.lower() == 'zico':
        the_nas_core = compute_zico.getzico(the_model, trainloader,lossfunc)

    elif args.zero_shot_score == 'Zen':
        the_nas_core_info = compute_zen_score.compute_nas_score(model=the_model, gpu=gpu,
                                                                resolution=args.input_image_size,
                                                                mixup_gamma=args.gamma, batch_size=args.batch_size,
                                                                repeat=1)
        the_nas_core = the_nas_core_info['avg_nas_score']
    elif args.zero_shot_score == 'TE-NAS':
        the_nas_core = compute_te_nas_score.compute_NTK_score(model=the_model, gpu=gpu,
                                                                resolution=args.input_image_size,
                                                                batch_size=args.batch_size)

    elif args.zero_shot_score == 'Syncflow':
        the_nas_core = compute_syncflow_score.do_compute_nas_score(model=the_model, gpu=gpu,
                                                                    resolution=args.input_image_size,
                                                                    batch_size=args.batch_size)

    elif args.zero_shot_score == 'GradNorm':
        the_nas_core = compute_gradnorm_score.compute_nas_score(model=the_model, gpu=gpu,
                                                                resolution=args.input_image_size,
                                                                batch_size=args.batch_size)

    elif args.zero_shot_score == 'Flops':
        the_nas_core = the_model.get_FLOPs(args.input_image_size)

    elif args.zero_shot_score.lower() == 'params':
        the_nas_core = the_model.get_model_size()

    elif args.zero_shot_score == 'Random':
        the_nas_core = np.random.randn()

    elif args.zero_shot_score == 'NASWOT':
        the_nas_core = compute_NASWOT_score.compute_nas_score(gpu=gpu, model=the_model,
                                                                resolution=args.input_image_size,
                                                                batch_size=args.batch_size)



    del the_model
    torch.cuda.empty_cache()
    return the_nas_core


def getmisc(args):
    if args.dataset == "cifar10":
        root = args.datapath
        imgsize=32
    elif args.dataset == "cifar100":
        root = args.datapath
        imgsize=32
    elif args.dataset.startswith("imagenet-1k"):
        root = args.datapath
        imgsize=224
    elif args.dataset.startswith("ImageNet16"):
        root = args.datapath
        imgsize=16
    
    
    train_data, test_data, xshape, class_num = datasets.get_datasets(args.dataset, root, 0)

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)
    return trainloader, testloader, xshape, class_num


def main(args, argv):

    gpu = args.gpu
    if gpu is not None:
        print(torch.cuda.device_count())
        torch.cuda.set_device('cuda:{}'.format(gpu))
        torch.backends.cudnn.benchmark = True
    print(args)
    trainloader, testloader, xshape, class_num = getmisc(args)
    trainbatches = []
    for batchid, batch in enumerate(trainloader):
        if batchid == args.maxbatch:
            break
        datax, datay = batch[0].cuda(), batch[1].cuda()
        trainbatches.append([datax, datay])
        
    best_structure_txt = os.path.join(args.save_dir, 'best_structure.txt')
    if os.path.isfile(best_structure_txt):
        print('skip ' + best_structure_txt)
        return None

    # load search space config .py file
    select_search_space = global_utils.load_py_module_from_path(args.search_space)

    # load masternet
    AnyPlainNet = Masternet.MasterNet

    masternet = AnyPlainNet(num_classes=args.num_classes, opt=args, argv=argv, no_create=True)
    initial_structure_str = str(masternet)

    popu_structure_list = []
    popu_zero_shot_score_list = []
    popu_latency_list = []
    search_time_list = []

    start_timer = time.time()
    lossfunc = nn.CrossEntropyLoss().cuda()
    loop_count = 0
    while loop_count < args.evolution_max_iter:
        # too many networks in the population pool, remove one with the smallest score
        while len(popu_structure_list) > args.population_size:
            min_zero_shot_score = min(popu_zero_shot_score_list)
            tmp_idx = popu_zero_shot_score_list.index(min_zero_shot_score)
            popu_zero_shot_score_list.pop(tmp_idx)
            popu_structure_list.pop(tmp_idx)
            popu_latency_list.pop(tmp_idx)
        pass

        # ----- generate a random structure ----- #
        if len(popu_structure_list) <= 10:
            random_structure_str = get_new_random_structure_str(
                AnyPlainNet=AnyPlainNet, structure_str=initial_structure_str, num_classes=args.num_classes,
                get_search_space_func=select_search_space.gen_search_space, num_replaces=1)
        else:
            tmp_idx = random.randint(0, len(popu_structure_list) - 1)
            tmp_random_structure_str = popu_structure_list[tmp_idx]
            random_structure_str = get_new_random_structure_str(
                AnyPlainNet=AnyPlainNet, structure_str=tmp_random_structure_str, num_classes=args.num_classes,
                get_search_space_func=select_search_space.gen_search_space, num_replaces=2)

        random_structure_str = get_splitted_structure_str(AnyPlainNet, random_structure_str,
                                                          num_classes=args.num_classes)

        the_model = None

        if args.max_layers is not None:
            if the_model is None:
                the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False)
            the_layers = the_model.get_num_layers()
            if args.max_layers < the_layers:
                continue

        if args.budget_model_size is not None:
            if the_model is None:
                the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False)
            the_model_size = the_model.get_model_size()
            if args.budget_model_size < the_model_size:
                continue

        if args.budget_flops is not None:
            if the_model is None:
                the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False)
            the_model_flops = the_model.get_FLOPs(args.input_image_size)
            if args.budget_flops < the_model_flops:
                continue

        the_latency = np.inf
        if args.budget_latency is not None:
            the_latency = get_latency(AnyPlainNet, random_structure_str, gpu, args)
            if args.budget_latency < the_latency:
                continue

        if loop_count >= 1 and loop_count % 100 == 0:
            max_score = max(popu_zero_shot_score_list)
            min_score = min(popu_zero_shot_score_list)
            elasp_time = time.time() - start_timer
            search_time = np.sum(search_time_list)
            logging.info(f'loop_count={loop_count}/{args.evolution_max_iter}, max_score={max_score:4g}, min_score={min_score:4g}, running_time={elasp_time/3600:4g}h, search_time={search_time/3600:4g}h')
        
        search_time_start = time.time()
        the_nas_core = compute_nas_score(AnyPlainNet, random_structure_str, gpu, args, trainbatches, lossfunc)
        search_time_list.append(time.time() - search_time_start)

        popu_structure_list.append(random_structure_str)
        popu_zero_shot_score_list.append(the_nas_core)
        popu_latency_list.append(the_latency)

        loop_count += 1

    return popu_structure_list, popu_zero_shot_score_list, popu_latency_list






if __name__ == '__main__':
    args = parse_cmd_options(sys.argv)
    log_fn = os.path.join(args.save_dir, 'evolution_search.log')
    global_utils.create_logging(log_fn)

    info = main(args, sys.argv)
    if info is None:
        exit()



    popu_structure_list, popu_zero_shot_score_list, popu_latency_list = info

    # export best structure
    best_score = max(popu_zero_shot_score_list)
    best_idx = popu_zero_shot_score_list.index(best_score)
    best_structure_str = popu_structure_list[best_idx]
    the_latency = popu_latency_list[best_idx]

    best_structure_txt = os.path.join(args.save_dir, 'best_structure.txt')
    global_utils.mkfilepath(best_structure_txt)
    with open(best_structure_txt, 'w') as fid:
        fid.write(best_structure_str)
    pass  # end with
