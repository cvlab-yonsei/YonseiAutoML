import os
import torch
import torch.nn as nn

class DSBN2d(nn.Module):
    def __init__(self, planes: int):
        super().__init__()
        self.num_features = planes
        self.BN_S = nn.BatchNorm2d(planes)
        self.BN_T = nn.BatchNorm2d(planes)
        self.mode = 1

    def forward(self, x):
        if not self.training:
            return self.BN_T(x)
        if self.mode == 1:
            return self.BN_S(x)
        elif self.mode == 2:
            return self.BN_T(x)
        else:
            bs = x.size(0)
            assert bs % 2 == 0
            split = torch.split(x, bs // 2, 0)
            out1 = self.BN_S(split[0].contiguous())
            out2 = self.BN_T(split[1].contiguous())
            return torch.cat((out1, out2), 0)

def convert_dsbn(module: nn.Module) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            m = DSBN2d(child.num_features)
            m.BN_S.load_state_dict(child.state_dict())
            m.BN_T.load_state_dict(child.state_dict())
            setattr(module, name, m)
        else:
            convert_dsbn(child)
    return module

def set_dsbn_mode(module: nn.Module, mode: int):
    for child in module.modules():
        if isinstance(child, DSBN2d):
            child.mode = int(mode)

def _select_device(device_str: str) -> str:
    if device_str:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_str)
    return "cuda" if torch.cuda.is_available() else "cpu"

def _build_model(model_name: str, num_classes: int) -> nn.Module:
    from torchvision.models import resnet18
    m = resnet18(num_classes=num_classes)
    return m

def convert_and_prepare(model_or_name, num_classes: int, device: str, use_aug: bool, mode=None) -> nn.Module:
    dev = _select_device(device)

    if isinstance(model_or_name, str):
        model = _build_model(model_or_name, num_classes).to(dev)
    else:
        model = model_or_name.to(dev)

    convert_dsbn(model)

    # 🔥 DSBN 레이어들은 새로 생성된 nn.Module들이라 GPU로 다시 올려야 함
    model = model.to(dev)

    if mode is None:
        mode = 2 if use_aug else 1
    set_dsbn_mode(model, mode)

    return model

