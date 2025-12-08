import torch
import torch.nn as nn
import torch.quantization
import torchvision
from torchvision.models import mobilenet_v2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from quantizer import quantize_model
from models.cnn import CNN
from models.layers import SearchSpaceNames
import matplotlib.pyplot as plt
import numpy as np

import logging
import argparse


def main(args):
    # 1. 데이터셋 준비 (ImageNet)
    # ImageNet 데이터셋의 경로 설정
    data_dir = "/dataset/ILSVRC2012/"  

    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=5, pin_memory=True)

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32, shuffle=True,
        num_workers=10, pin_memory=True)

    print(f'data 개수: {len(val_dataset)}')

    num_samples_per_class = {}
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            for k in range(images.size(0)):
                label = labels[k].item()
                if label not in num_samples_per_class:
                    num_samples_per_class[label] = 0
                num_samples_per_class[label] += 1
    print("Number of samples per class in test dataset: ", num_samples_per_class)

    dataiter = iter(val_loader)
    images, labels = dataiter.next()

    def denormalize(image):
        mean = (0.4914, 0.4822, 0.4465)
        variane = (0.2023, 0.1994, 0.2010)
        image = image.numpy().astype(dtype=np.float32)
        for i in range(3):
            image[i] = image[i] * variane[i] + mean[i]
            
        return np.transpose(image, (1, 2, 0))

    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(denormalize(images[i]))

        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.savefig('imagenet.png')

    images, _ = next(iter(val_loader))
    images_for_test = images.cuda() 




    # with torch.no_grad():
    #     correct_classified = 0
    #     total = 0
    #     for i, (images, target) in enumerate(val_loader):
    #         images = images.cuda()
    #         labels = target.cuda()

    #         # compute output
    #         output = model(images)

    #         pred = model(images)
    #         _, predicted = torch.max(pred.data, 1)
    #         total += pred.size(0)
    #         correct_classified += (predicted == labels).sum().item()
            
    #     test_acc = correct_classified/total*100
    #     print("Test accuracy: {}%".format(test_acc))

    # print("===========================================================")

    def load_arch_from_txt(path):
        with open(path, "r") as f:
            content = f.read().strip()

        # "2, 2, 6, ..." → ['2','2','6', ...] → [2,2,6,...]
        nums = [int(x.strip()) for x in content.split(",")]

        return tuple(nums)

    print(f"============================={'Our Model'}==============================")

    search_space = SearchSpaceNames['proxyless']
    arch = (2, 2, 6, 0, 0, 4, 0, 2, 4, 0, 2, 0, 0, 2, 0, 6, 3, 6, 6, 6, 1)
    model = CNN(search_space, arch, 0).cuda()

    # FLOPs 계산
    # print('Our searched model')
    # flops = FlopCountAnalysis(model, input_tensor)
    # print(f"FLOPs: {flops.total()}")

    # ckpt = torch.load('epoch-241.pt')['state_dict']
    ckpt = torch.load(args.ckpt, map_location="cuda")['state_dict']
    model.load_state_dict(ckpt)
    model = quantize_model(model, args.bits).cuda()

    model.eval()

    # Quantized checkpoint 저장 경로 생성
    ckpt_dir = os.path.dirname(args.ckpt)
    ckpt_name = os.path.basename(args.ckpt).replace(".pt", f"_quant{args.bits}.pt")
    save_path = os.path.join(ckpt_dir, ckpt_name)

    # 저장 (state_dict 기준)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "quant_bits": args.bits,
            "arch": arch,   # arch 변수 이미 위에서 존재함
        },
        save_path,
    )

    # 저장 경로 출력
    print(f"\n✅ Quantized checkpoint saved to:\n{save_path}\n")

    # MAC, _ = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    # print(f"MAC: {MAC}")



    # For quantization
    with torch.no_grad():
        images, _ = next(iter(train_loader))
        images = images.cuda()
        model(images)

    # For time measurement


    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # images, _ = next(iter(val_loader))
    # images = images.cuda() 

    # Warm up
    for _ in range(10):
        _ = model(images_for_test)

    with torch.no_grad():
        start_event.record()
        for _ in range(10):          
            model(images_for_test)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        our_searched_model_elapsed_time = elapsed_time
        print("Time for 10batch inference: {}ms".format(elapsed_time))

    with torch.no_grad():
        correct_classified = 0
        total = 0
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            labels = target.cuda()

            # compute output
            output = model(images)

            pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            total += pred.size(0)
            correct_classified += (predicted == labels).sum().item()
            
        test_acc = correct_classified/total*100
        print("Test accuracy: {}%".format(test_acc))


    print(f"============================={'MobileNet'}==============================")

    model = mobilenet_v2(pretrained=True).cuda()




    print('MobileNet')
    # FLOPs 계산
    # flops = FlopCountAnalysis(model, input_tensor)
    # print(f"FLOPs: {flops.total()}")

    model = quantize_model(model, 8).cuda()

    model.eval()



    # For quantization
    with torch.no_grad():
        images, _ = next(iter(train_loader))
        images = images.cuda()
        model(images)

    # For time measurement

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)


    # Warm up
    for _ in range(10):
        _ = model(images_for_test)

    with torch.no_grad():
        start_event.record()
        for _ in range(10):          
            model(images_for_test)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        mobilenet_elapsed_time = elapsed_time
        print("Time for 10batch inference: {}ms".format(elapsed_time))

    with torch.no_grad():
        correct_classified = 0
        total = 0
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            labels = target.cuda()

            # compute output
            output = model(images)

            pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            total += pred.size(0)
            correct_classified += (predicted == labels).sum().item()
            
        test_acc = correct_classified/total*100
        print("Test accuracy: {}%".format(test_acc))

    time_reduction_rate = (mobilenet_elapsed_time-our_searched_model_elapsed_time) / mobilenet_elapsed_time * 100
    print(f'Time reduction rate: {time_reduction_rate}%')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--arch_file", type=str, required=True,
                        help="Path to architecture txt file")

    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint file (.pt)")
    
    parser.add_argument("--bits", type=int, default=8)


    args = parser.parse_args()

    main(args)