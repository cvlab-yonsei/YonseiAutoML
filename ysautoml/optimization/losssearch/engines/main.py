import argparse
import torch
import torchvision
import numpy as np
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from model import resnet20
from loss import custom_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr_model", type=float, default=0.1)
    parser.add_argument("--lr_loss", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=0.0005)
    parser.add_argument("--save_dir", type=str, default="./logs/losssearch")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[LOSSSEARCH] Training on {device}")

    # Dataset
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    dataset_path = "./dataset"
    train_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Model
    model = resnet20().to(device)
    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, momentum=args.momentum, weight_decay=args.wd)

    # Custom Loss
    criterion = custom_loss().to(device)
    optimizer_loss = torch.optim.SGD(criterion.parameters(), lr=args.lr_loss, momentum=args.momentum)

    scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_model, T_max=args.epochs, eta_min=0)
    scheduler_loss = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_loss, T_max=args.epochs, eta_min=0)

    print("[LOSSSEARCH] Start training...")

    for ep in range(args.epochs):
        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer_model.zero_grad()
            optimizer_loss.zero_grad()

            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer_model.step()

            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total * 100

        if ep % 10 == 0 and ep != 0:
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer_model.zero_grad()
                optimizer_loss.zero_grad()
                preds = model(images)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer_loss.step()
            print(f"[LOSSSEARCH] Epoch {ep}: Updated θ → {criterion.theta}")

        scheduler_model.step()
        scheduler_loss.step()

        print(f"[LOSSSEARCH] Epoch {ep}: Train Acc {train_acc:.2f}%")

    print("[LOSSSEARCH] Training complete!")


if __name__ == "__main__":
    main()
