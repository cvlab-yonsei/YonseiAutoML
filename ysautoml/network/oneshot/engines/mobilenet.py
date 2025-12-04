import torch
import torch.nn as nn
import torch.quantization
import torchvision
from torchvision.models import mobilenet_v2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from quantizer import quantize_model
from fvcore.nn import FlopCountAnalysis

import logging
logging.getLogger("fvcore").setLevel(logging.ERROR)

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

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=32, shuffle=False,
    num_workers=10, pin_memory=True)

model = mobilenet_v2(pretrained=True).cuda()


input_tensor = torch.randn(1, 3, 224, 224).cuda()

print('MonileNet')
# FLOPs 계산
flops = FlopCountAnalysis(model, input_tensor)
print(f"FLOPs: {flops.total()}")

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

images, _ = next(iter(val_loader))
images = images.cuda() 

# Warm up
for _ in range(10):
    _ = model(images)

with torch.no_grad():
    start_event.record()
    for _ in range(10):          
        model(images)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    print("Time for 1batch inference: {}ms".format(elapsed_time/10))

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

print("===========================================================")