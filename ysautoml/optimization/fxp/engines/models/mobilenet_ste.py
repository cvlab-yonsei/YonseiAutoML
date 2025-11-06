import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import re, os
from .modules import DSQConv_a, DSQConv_8bit, DSQLinear, DSQConv_a_mobile


__all__ = ['MobileNetV2', 'mobilenet_ste']

# ----------------------------------------------
# 기본 MobileNetV2 (원본 유지)
# ----------------------------------------------

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None, initial=False, symmetric=False):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if initial:
            QInput = False
        else:
            QInput = True

        if initial or (groups > 1):
            super(ConvBNReLU, self).__init__(
                nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
                norm_layer(out_planes),
                nn.ReLU6(inplace=True)
            )
        else:
            super(ConvBNReLU, self).__init__(
                DSQConv_a_mobile(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False, QInput=QInput, symmetric=symmetric),
                norm_layer(out_planes),
                nn.ReLU6(inplace=True)
            )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None, first_layer=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        symmetric = not first_layer
        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, symmetric=symmetric))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer, symmetric=False),
            # pw-linear
            DSQConv_a_mobile(hidden_dim, oup, 1, 1, 0, bias=False, symmetric=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, num_bit=4, inverted_residual_setting=None, round_nearest=8, block=None, norm_layer=None):
        super(MobileNetV2, self).__init__()
        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer, initial=True)]
        first_layer = True
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, first_layer=first_layer))
                input_channel = output_channel
                first_layer = False

        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, symmetric=True))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), DSQLinear(self.last_channel, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return {"out": x}


# ----------------------------------------------
# QuantPlainNet: CIFAR100 대응 + DSQConv 사용
# ----------------------------------------------

class QuantPlainNet(nn.Module):
    def __init__(self, arch_path, num_classes=100):
        super().__init__()
        assert os.path.exists(arch_path), f"Architecture file not found: {arch_path}"
        with open(arch_path, "r") as f:
            arch_str = f.read().strip()

        pattern = r"(Super\w+)\((.*?)\)"
        blocks = re.findall(pattern, arch_str)
        layers = []
        in_ch = 3

        for block_type, args in blocks:
            args = list(map(int, args.split(",")))

            # ---- SuperConvK3BNRELU ----
            if "SuperConvK3BNRELU" in block_type:
                _, out_ch, stride, _ = args
                stride = 1  # CIFAR용: stride 줄임
                layers.append(nn.Sequential(
                    DSQConv_a_mobile(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ))
                in_ch = out_ch

            # ---- SuperResIDWE ----
            elif "SuperResIDWE" in block_type:
                _, out_ch, stride, hidden, exp = args
                stride = 1  # CIFAR용 stride 고정
                expand_ch = int(hidden)
                layers.append(nn.Sequential(
                    DSQConv_a_mobile(in_ch, expand_ch, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(expand_ch),
                    nn.ReLU(inplace=True),
                    DSQConv_a_mobile(expand_ch, expand_ch, kernel_size=3, stride=stride, padding=1, groups=expand_ch, bias=False),
                    nn.BatchNorm2d(expand_ch),
                    nn.ReLU(inplace=True),
                    DSQConv_a_mobile(expand_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_ch),
                ))
                in_ch = out_ch

            # ---- SuperConvK1BNRELU ----
            elif "SuperConvK1BNRELU" in block_type:
                _, out_ch, _, _ = args
                layers.append(nn.Sequential(
                    DSQConv_a_mobile(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ))
                in_ch = out_ch

        # ---- Classifier for CIFAR100 ----
        layers.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 전역 평균 풀링
            nn.Flatten(),
            DSQLinear(in_ch, num_classes)
        ))

        self.features = nn.Sequential(*layers)

        # 안정화: scale 조정
        for m in self.modules():
            if isinstance(m, DSQConv_a_mobile) and hasattr(m, 'sW'):
                m.sW.data *= 0.1

    def forward(self, x):
        x = self.features(x)
        return {"out": x}


# ----------------------------------------------
# mobilenet_ste unified entry
# ----------------------------------------------

def mobilenet_ste(pretrained=False, progress=True, arch_path=None, num_classes=100, **kwargs):
    """
    arch_path가 주어지면 QuantPlainNet(CIFAR용 DSQConv 구조) 사용
    그렇지 않으면 기본 MobileNetV2 사용
    """
    if arch_path is not None:
        print(f"[mobilenet_ste] Loading structure from {arch_path}")
        return QuantPlainNet(arch_path, num_classes=num_classes)
    else:
        return MobileNetV2(num_classes=num_classes, **kwargs)
