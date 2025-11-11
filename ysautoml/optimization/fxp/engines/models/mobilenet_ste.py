import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import os, re
from .modules import DSQConv_a, DSQConv_8bit, DSQLinear, DSQConv_a_mobile


__all__ = ['MobileNetV2', 'mobilenet_ste']


# -----------------------------
# 기본 MobileNetV2 구성 요소
# -----------------------------
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 groups=1, norm_layer=None, initial=False, symmetric=False):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if initial:
            QInput = False
        else:
            QInput = True

        if initial or (groups > 1):
            super().__init__(
                nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                          padding, groups=groups, bias=False),
                norm_layer(out_planes),
                nn.ReLU6(inplace=True),
            )
        else:
            super().__init__(
                DSQConv_a_mobile(in_planes, out_planes, kernel_size, stride,
                                 padding, groups=groups, bias=False,
                                 QInput=QInput, symmetric=symmetric),
                norm_layer(out_planes),
                nn.ReLU6(inplace=True),
            )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None,
                 first_layer=False):
        super().__init__()
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
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1,
                                     norm_layer=norm_layer, symmetric=symmetric))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride,
                       groups=hidden_dim, norm_layer=norm_layer, symmetric=False),
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
    def __init__(self, num_classes=1000, width_mult=1.0, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        block = InvertedResidual

        input_channel = 32
        last_channel = 1280
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

        features = [ConvBNReLU(3, input_channel, stride=2,
                               norm_layer=norm_layer, initial=True)]
        first_layer = True
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride,
                          expand_ratio=t, norm_layer=norm_layer,
                          first_layer=first_layer)
                )
                input_channel = output_channel
                first_layer = False
        features.append(
            ConvBNReLU(input_channel, self.last_channel, kernel_size=1,
                       norm_layer=norm_layer, symmetric=True)
        )

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            DSQLinear(self.last_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return {"out": x}


# -----------------------------
# PlainNet 구조 파서 (ImageNet 기준)
# -----------------------------
class QuantPlainNet(nn.Module):
    """
    best_structure.txt 로부터 구조를 파싱해 DSQConv 기반으로 네트워크 구성
    (stride, channel 등 ImageNet 기준 유지)
    """
    def __init__(self, arch_path, num_classes=1000):
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

            if "SuperConvK3BNRELU" in block_type:
                _, out_ch, stride, _ = args
                layers.append(nn.Sequential(
                    DSQConv_a_mobile(in_ch, out_ch, kernel_size=3,
                                     stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ))
                in_ch = out_ch

            elif "SuperResIDWE" in block_type:
                _, out_ch, stride, hidden, exp = args
                expand_ch = int(hidden)
                layers.append(nn.Sequential(
                    DSQConv_a_mobile(in_ch, expand_ch, kernel_size=1,
                                     stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(expand_ch),
                    nn.ReLU(inplace=True),
                    DSQConv_a_mobile(expand_ch, expand_ch, kernel_size=3,
                                     stride=stride, padding=1, groups=expand_ch, bias=False),
                    nn.BatchNorm2d(expand_ch),
                    nn.ReLU(inplace=True),
                    DSQConv_a_mobile(expand_ch, out_ch, kernel_size=1,
                                     stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_ch),
                ))
                in_ch = out_ch

            elif "SuperConvK1BNRELU" in block_type:
                _, out_ch, _, _ = args
                layers.append(nn.Sequential(
                    DSQConv_a_mobile(in_ch, out_ch, kernel_size=1,
                                     stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ))
                in_ch = out_ch

        layers.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            DSQLinear(in_ch, num_classes)
        ))

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return {"out": self.features(x)}


# -----------------------------
# unified mobilenet_ste entry
# -----------------------------
def mobilenet_ste(pretrained=False, progress=True, arch_path=None, num_classes=1000, **kwargs):
    """
    if arch_path is provided -> QuantPlainNet(DSQConv 기반)
    else -> default MobileNetV2
    """
    if arch_path is not None:
        print(f"[mobilenet_ste] Loading structure from {arch_path}")
        return QuantPlainNet(arch_path, num_classes=num_classes)
    else:
        return MobileNetV2(num_classes=num_classes, **kwargs)
