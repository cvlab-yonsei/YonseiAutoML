import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

# <-- 기존 utils.py의 함수들을 상대경로로 import
from ..utils import (
    get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset,
    get_daparam, match_loss, get_time, TensorDataset, epoch,
    DiffAugment, ParamDiffAug, BatchAug
)

def _select_device(device_str: str) -> str:
    """
    device_str: '0', '1,2' 등 환경 변수 값으로 들어오는 문자열을 허용.
    CUDA 가용 시 'cuda' 반환, 아니면 'cpu'.
    """
    if device_str is not None and len(str(device_str)) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_str)
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def run(args):
    """
    Dataset Condensation / DSA 실행 엔진.
    - args는 parser에서 만들어진 Namespace 형태(또는 동등 객체)여야 함.
    - wandb 의존 및 argparse 진입부는 제거.
    - 실행 후 최종 accuracy 및 저장 경로 등을 dict로 반환.

    필수 args 예시:
        method (str): 'DSA' or 'DC'
        dataset (str): 'CIFAR10' ...
        model (str): 'ConvNet' ...
        ipc (int)
        eval_mode (str)
        num_exp (int)
        num_eval (int)
        epoch_eval_train (int)
        Iteration (int)
        lr_img (float)
        lr_net (float)
        batch_real (int)
        batch_train (int)
        init (str) : 'noise' or 'real'
        dsa_strategy (str)
        data_path (str)
        dis_metric (str)
        device (str) : '0' 등
        run_name (str)
        run_tags (str|None)
    """
    # loops
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    if args.dataset == 'CIFAR100' and args.ipc == 50:
        args.inner_loop = 10
        args.outer_loop = 10

    # device & DSA params
    args.device = _select_device(getattr(args, "device", "0"))
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    # data path check
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Wrong data directory: {args.data_path}")

    # save path
    args.save_path = os.path.join('./logs', args.run_name)
    os.makedirs(args.save_path, exist_ok=True)

    # dataset & eval pool
    channel, im_size, num_classes, _, mean, std, dst_train, _, testloader = \
        get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    # record
    accs_all_exps = {key: [] for key in model_eval_pool}
    data_save = []
    dsa_params = args.dsa_param  # keep for logging if needed

    for exp in range(args.num_exp):
        print(f'\n================== Exp {exp} ==================\n ')
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        # organize real dataset
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        indices_class = [[] for _ in range(num_classes)]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0)
        labels_all = torch.tensor(labels_all, dtype=torch.long)

        for c in range(num_classes):
            print(f'class c = {c}: {len(indices_class[c])} real images')

        def get_images(c, n):  # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle].to(args.device)

        for ch in range(channel):
            print(f'real images channel {ch}, mean = {torch.mean(images_all[:, ch]):.4f}, '
                  f'std = {torch.std(images_all[:, ch]):.4f}')

        # initialize the synthetic data
        image_syn = torch.randn(
            size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
            dtype=torch.float, requires_grad=True, device=args.device
        )
        label_syn = torch.tensor(
            [np.ones(args.ipc) * i for i in range(num_classes)],
            dtype=torch.long, requires_grad=False, device=args.device
        ).view(-1)

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = \
                    get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')

        # training
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5)
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        print(f'{get_time()} training begins')

        for it in range(args.Iteration + 1):

            # evaluation at last iter
            if it == args.Iteration:
                for model_eval in model_eval_pool:
                    print('-------------------------')
                    print(f'Evaluation\nmodel_train = {args.model}, model_eval = {model_eval}, iteration = {it}')
                    if args.dsa:
                        args.epoch_eval_train = 1000
                        args.dc_aug_param = None
                        print('DSA augmentation strategy: \n', args.dsa_strategy)
                        print('DSA augmentation parameters: \n', dsa_params.__dict__)
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc)
                        print('DC augmentation parameters: \n', args.dc_aug_param)

                    if args.dsa or (args.dc_aug_param and args.dc_aug_param.get('strategy', 'none') != 'none'):
                        args.epoch_eval_train = 1000
                    else:
                        args.epoch_eval_train = 300

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                        image_syn_eval = copy.deepcopy(image_syn.detach())
                        label_syn_eval = copy.deepcopy(label_syn.detach())
                        _, acc_train, acc_test = evaluate_synset(
                            it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args
                        )
                        accs.append(acc_test)
                    print(f'Evaluate {len(accs)} random {model_eval}, mean = {np.mean(accs):.4f} '
                          f'std = {np.std(accs):.4f}\n-------------------------')
                    accs_all_exps[model_eval] += accs

                # visualize & save
                save_name = os.path.join(
                    args.save_path,
                    f'vis_{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc_exp{exp}_iter{it}.png'
                )
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis < 0] = 0.0
                image_syn_vis[image_syn_vis > 1] = 1.0
                save_image(image_syn_vis, save_name, nrow=10)
                break  # end training loop

            # Train synthetic data (outer/inner loops)
            net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
            optimizer_net.zero_grad()
            loss_avg = 0.0
            args.dc_aug_param = None  # mute DC aug in inner-loop to match DC paper

            for ol in range(args.outer_loop):

                # Freeze BN running stats using real data
                BN_flag = any('BatchNorm' in m._get_name() for m in net.modules())
                if BN_flag:
                    BNSizePC = 16
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train()
                    _ = net(img_real)  # update running stats
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():
                            module.eval()

                # update synthetic data
                optimizer_img.zero_grad()
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape(
                        (args.ipc, channel, im_size[0], im_size[1])
                    )
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                    # FYI: concatenate flipped images
                    img_syn, lab_syn = BatchAug(img_syn, lab_syn)

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = [_.detach().clone() for _ in gw_real]

                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    loss = match_loss(gw_syn, gw_real, args)
                    loss.backward()
                    loss_avg += loss.item()

                optimizer_img.step()

                if ol == args.outer_loop - 1:
                    break

                # update network using current synthetic set
                image_syn_train = copy.deepcopy(image_syn.detach())
                label_syn_train = copy.deepcopy(label_syn.detach())
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(
                    dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0
                )
                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args,
                          aug=True if args.dsa else False, batch_aug=True)

            loss_avg /= (num_classes * args.outer_loop)

            if it % 10 == 0:
                print(f'{get_time()} iter = {it:04d}, loss = {loss_avg:.4f}')

            if it == args.Iteration:
                data_save.append([copy.deepcopy(image_syn.detach().cpu()),
                                  copy.deepcopy(label_syn.detach().cpu())])
                torch.save(
                    {'data': data_save, 'accs_all_exps': accs_all_exps},
                    os.path.join(args.save_path,
                                 f'res_{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc.pt')
                )

    # Final results
    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print("Accuracy")
        print('Run %d experiments, train on %s, evaluate %d random %s, '
              'mean  = %.2f%%  std = %.2f%%' % (
                  args.num_exp, args.model, len(accs), key,
                  np.mean(accs) * 100, np.std(accs) * 100)
              )

    return {
        "save_path": args.save_path,
        "accs_all_exps": accs_all_exps,
        "eval_pool": model_eval_pool,
        "num_exp": args.num_exp,
    }
