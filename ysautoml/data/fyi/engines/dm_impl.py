import os
import time
import copy
import numpy as np
import torch
from torchvision.utils import save_image

from ..utils import (
    get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset,
    get_daparam, get_time, TensorDataset, DiffAugment, ParamDiffAug, BatchAug
)

def _select_device(device_str: str) -> str:
    """'0', '1,2' 같은 문자열을 허용. CUDA 가능 시 'cuda', 아니면 'cpu'."""
    if device_str is not None and len(str(device_str)) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_str)
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def run(args):
    """
    Distillation Matching (DM) 실행 엔진.
    - args는 parser.Namespace와 동등한 객체여야 함.
    - 필수 인자: dataset, model, ipc, eval_mode, num_exp, num_eval, epoch_eval_train,
                 Iteration, lr_img, batch_real, batch_train, init, dsa_strategy,
                 data_path, device, run_name, (선택)run_tags
    """
    # 고정/초기 설정
    args.method = 'DM'
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = _select_device(getattr(args, "device", "0"))
    args.dsa_param = ParamDiffAug()
    args.dsa = False if getattr(args, "dsa_strategy", "none") in ['none', 'None'] else True

    # 데이터 경로/저장 경로
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Wrong data directory: {args.data_path}")
    args.save_path = os.path.join('./logs', args.run_name)
    os.makedirs(args.save_path, exist_ok=True)

    # 데이터셋 & 평가 풀
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = \
        get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    # 기록 구조
    accs_all_exps = {key: [] for key in model_eval_pool}
    data_save = []

    for exp in range(args.num_exp):
        print(f'\n================== Exp {exp} ==================\n ')
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        # 실데이터 정리
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        indices_class = [[] for _ in range(num_classes)]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0)
        labels_all = torch.tensor(labels_all, dtype=torch.long)

        for c in range(num_classes):
            print(f'class c = {c}: {len(indices_class[c])} real images')

        def get_images(c, n):
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle].to(args.device)

        for ch in range(channel):
            print(f'real images channel {ch}, mean = {torch.mean(images_all[:, ch]):.4f}, '
                  f'std = {torch.std(images_all[:, ch]):.4f}')

        # 합성 데이터 초기화
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

        # 학습
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5)
        optimizer_img.zero_grad()
        print(f'{get_time()} training begins')

        for it in range(args.Iteration + 1):

            # 마지막 이터레이션에서 평가
            if it == args.Iteration:
                for model_eval in model_eval_pool:
                    print('-------------------------')
                    print(f'Evaluation\nmodel_train = {args.model}, model_eval = {model_eval}, iteration = {it}')
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

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

                # 시각화 저장
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
                break  # 학습 루프 종료

            # 합성데이터 업데이트(임베딩 매칭)
            net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            net.train()
            for p in net.parameters():
                p.requires_grad = False

            embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed

            loss_avg = 0.0

            if 'BN' not in args.model:  # ConvNet 가정
                optimizer_img.zero_grad()
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape(
                        (args.ipc, channel, im_size[0], im_size[1])
                    )

                    # FYI: 합성 이미지에 flip 등 batch augment
                    img_syn, _ = BatchAug(img_syn, None)

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)
                    loss = torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)

                    loss.backward()
                    loss_avg += loss.item()
                optimizer_img.step()

            loss_avg /= (num_classes)

            if it % 10 == 0:
                print(f'{get_time()} iter = {it:05d}, loss = {loss_avg:.4f}')

            if it == args.Iteration:
                data_save.append([copy.deepcopy(image_syn.detach().cpu()),
                                  copy.deepcopy(label_syn.detach().cpu())])
                torch.save(
                    {'data': data_save, 'accs_all_exps': accs_all_exps},
                    os.path.join(args.save_path,
                                 f'res_{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc.pt')
                )

    # 최종 결과 출력
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
