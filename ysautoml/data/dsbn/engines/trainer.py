import torch
import torch.nn as nn
import torch.optim as optim
from .dsbn_impl import set_dsbn_mode

def train_with_dsbn(model, train_loader_source, train_loader_target=None,
                    epochs=1, lr=0.1, mixed_batch=False, device="cuda",
                    log_interval=10):
    """
    Train a DSBN-converted model.

    Args:
        model: DSBN-converted nn.Module
        train_loader_source: DataLoader for source domain
        train_loader_target: DataLoader for target domain (if not mixed)
        epochs: training epochs
        lr: learning rate
        mixed_batch: if True, expects loader to yield mixed source+target batches
        device: "cuda" or "cpu"
        log_interval: steps마다 로그 출력

    Returns:
        dict with:
            - "logs": list of (epoch, step, loss) tuples
            - "final_acc": float (최종 source loader accuracy)
            - "state_dict": model.state_dict() (마지막 학습 상태)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    logs = []
    model.train()

    for epoch in range(2):
        if mixed_batch:
            # 한 배치에 소스+타깃 섞여 들어온 경우
            for step, (imgs, labels) in enumerate(train_loader_source):
                if step >= 50:   # ✅ step 50까지만 실행
                    break   
                set_dsbn_mode(model, 3)
                imgs, labels = imgs.to(device), labels.to(device)

                logits = model(imgs)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % log_interval == 0:
                    print(f"[Epoch {epoch+1}/{epochs}][Step {step}] Loss={loss.item():.4f}", flush=True)
                logs.append((epoch, step, loss.item()))

        else:
            # 소스/타깃 따로 도는 경우
            for step, ((imgs_s, labels_s), (imgs_t, labels_t)) in enumerate(zip(train_loader_source, train_loader_target)):
                # 1) 소스 배치
                set_dsbn_mode(model, 1)
                imgs_s, labels_s = imgs_s.to(device), labels_s.to(device)
                logits_s = model(imgs_s)
                loss_s = criterion(logits_s, labels_s)
                optimizer.zero_grad()
                loss_s.backward()
                optimizer.step()

                # 2) 타깃 배치
                set_dsbn_mode(model, 2)
                imgs_t, labels_t = imgs_t.to(device), labels_t.to(device)
                logits_t = model(imgs_t)
                loss_t = criterion(logits_t, labels_t)
                optimizer.zero_grad()
                loss_t.backward()
                optimizer.step()

                if step % log_interval == 0:
                    print(f"[Epoch {epoch+1}/{epochs}][Step {step}] LossS={loss_s.item():.4f} LossT={loss_t.item():.4f}", flush=True)
                logs.append((epoch, step, (loss_s.item(), loss_t.item())))

        print(f"[Epoch {epoch+1}/{epochs}] Done", flush=True)

    # --- 최종 Accuracy 측정 (source domain 기준) ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for step, (imgs, labels) in enumerate(train_loader_source):
            if step >= 50:   # ✅ 평가도 50 step까지만 실행
                break
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    final_acc = correct / total if total > 0 else 0.0
    print(f"[Final Accuracy] {final_acc*100:.2f}%", flush=True)

    return {
        "logs": logs,
        "final_acc": final_acc,
        "state_dict": model.state_dict()
    }
