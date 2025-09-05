from typing import Optional
import torch.nn as nn
from .engines.dsbn_impl import convert_and_prepare, set_dsbn_mode as _set_mode
from .engines.trainer import train_with_dsbn

def convert_and_wrap(model_or_name, dataset="CIFAR10", num_classes=10,
                     use_aug=False, mode: Optional[int]=None, device="0",
                     export_path: Optional[str]=None) -> nn.Module:
    model = convert_and_prepare(model_or_name, num_classes, device, use_aug, mode)
    if export_path:
        import torch
        from pathlib import Path
        Path(export_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), export_path)
    return model

def set_dsbn_mode(model: nn.Module, mode: int):
    _set_mode(model, mode)

def train_with_dsbn_api(model, source_loader, target_loader=None,
                        epochs=1, lr=0.1, mixed_batch=False, device="cuda"):
    return train_with_dsbn(model, source_loader, target_loader,
                           epochs=epochs, lr=lr, mixed_batch=mixed_batch, device=device)
