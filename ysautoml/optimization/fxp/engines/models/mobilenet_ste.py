import os
import re
import math
import json
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import torch.utils.model_zoo as model_zoo

from .modules import DSQConv_a, DSQConv_8bit, DSQLinear, DSQConv_a_mobile

__all__ = ['MobileNetV2', 'mobilenet_ste']

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    """
    - initial=True 이거나 depthwise(groups>1)면 일반 Conv2d 사용
    - 그 외에는 DSQConv_a_mobile 사용
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1,
                 norm_layer=None, initial=False, symmetric=False):
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
                DSQConv_a_mobile(in_planes, out_planes, kernel_size, stride, padding, groups=groups,
                                 bias=False, QInput=QInput, symmetric=symmetric),
                norm_layer(out_planes),
                nn.ReLU6(inplace=True)
            )


class InvertedResidualExplicit(nn.Module):
    """
    Inverted Residual block (MobileNetV2) 변형:
    - hidden_dim을 ratio로부터 계산하지 않고, 아키텍처 문자열의 명시값을 그대로 사용
    - depthwise: K=3, groups=hidden_dim
    - pw-linear: DSQConv_a_mobile 사용 (symmetric=False)
    - 첫 stage는 first_layer 플래그를 받아 symmetry 제어(원 코드 호환)
    """
    def __init__(self, inp, oup, stride, hidden_dim, norm_layer=None, first_layer=False):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.use_res_connect = (self.stride == 1 and inp == oup)

        # 원 코드 로직과 호환: 첫 블록만 symmetric=False
        symmetric = not first_layer

        layers = []
        if hidden_dim != inp:
            # PW (1x1)
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, symmetric=symmetric))
        # DW (3x3)
        layers.append(ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                                 groups=hidden_dim, norm_layer=norm_layer, symmetric=False))
        # PW-Linear
        layers.append(DSQConv_a_mobile(hidden_dim, oup, 1, 1, 0, bias=False, symmetric=False))
        layers.append(norm_layer(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    기본 MobileNetV2 (arch_path 미지정 시 사용).
    """
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 num_bit=4,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidualExplicit  # 내부적으로 hidden_dim 직접 지정 형태 사용
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s  (여기서는 hidden_dim을 ratio로 계산해야 하지만, Explicit 블록과 호환을 위해 근사 적용)
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer, initial=True)]

        first_layer = True
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                # hidden_dim을 ratio로 만들되, Explicit 블록 시그니처에 맞게 명시 전달
                hidden_dim = int(round(input_channel * t))
                features.append(InvertedResidualExplicit(input_channel, output_channel, stride,
                                                         hidden_dim=hidden_dim,
                                                         norm_layer=norm_layer,
                                                         first_layer=first_layer))
                input_channel = output_channel
                first_layer = False

        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1,
                                   norm_layer=norm_layer, symmetric=True))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        ret_dict = dict()
        ret_dict['out'] = self._forward_impl(x)
        return ret_dict


# -----------------------------
# Parser & Builder for arch_path
# -----------------------------

_ARCH_TOKEN_RE = re.compile(r'([A-Za-z0-9_]+)\(([^)]*)\)')

def _parse_arch_string(s: str):
    """
    'SuperConvK3BNRELU(3,8,2,1)SuperResIDWE4K3(16,64,2,40,5)...' 를
    [ (name, [args...]), ... ] 형태로 파싱
    """
    s = s.replace('\n', '').replace(' ', '')
    tokens = []
    for m in _ARCH_TOKEN_RE.finditer(s):
        name = m.group(1)
        args = m.group(2)
        if args == '':
            arg_list = []
        else:
            arg_list = [int(x) for x in args.split(',')]
        tokens.append((name, arg_list))
    if not tokens:
        raise ValueError('Failed to parse architecture string. Check your arch file.')
    return tokens


class MobileNetFromArch(nn.Module):
    """
    arch_path로부터 파싱된 토큰을 기반으로 features를 구성.
    - SuperConvK3BNRELU(in_c,out_c,stride,n)
    - SuperResIDWE{t}K3(in_c,out_c,stride,hidden_c,n)
    - SuperConvK1BNRELU(in_c,out_c,stride,n)
    마지막 1x1 Conv의 out_c를 last_channel로 사용.
    """
    def __init__(self, tokens, num_classes=1000, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        features = []
        in_ch_tracker = None
        last_out_ch = None

        first_layer = True

        for name, args in tokens:
            if name.startswith('SuperConvK3BNRELU'):
                # (in_c, out_c, stride, n)
                if len(args) != 4:
                    raise ValueError(f'{name} expects 4 args, got {len(args)}: {args}')
                in_c, out_c, stride, n = args
                if n != 1:
                    # 필요하면 여기서 반복 구현 가능. 현재 구조 예시는 n==1
                    raise NotImplementedError(f'{name} with n={n} not supported yet.')
                features.append(
                    ConvBNReLU(in_c, out_c, kernel_size=3, stride=stride,
                               norm_layer=norm_layer, initial=True)  # 첫 레이어는 initial=True
                )
                in_ch_tracker = out_c
                last_out_ch = out_c
                first_layer = False

            elif name.startswith('SuperResIDWE') and name.endswith('K3'):
                # name: SuperResIDWE{t}K3
                # args: (in_c, out_c, stride, hidden_c, n)
                if len(args) != 5:
                    raise ValueError(f'{name} expects 5 args, got {len(args)}: {args}')
                in_c, out_c, stride, hidden_c, n = args

                # 안전 체크: 입력 채널 일치 보장 (파일이 다르면 강제로 맞추진 않음)
                if in_ch_tracker is not None and in_c != in_ch_tracker:
                    # 필요 시 여기서 자동 보정 레이어를 넣을 수도 있지만, 오류로 처리
                    raise ValueError(f'Input channel mismatch at {name}: file says {in_c}, but current={in_ch_tracker}')

                for i in range(n):
                    s = stride if i == 0 else 1
                    features.append(
                        InvertedResidualExplicit(in_c, out_c, s, hidden_dim=hidden_c,
                                                 norm_layer=norm_layer, first_layer=first_layer)
                    )
                    in_c = out_c
                    in_ch_tracker = out_c
                    last_out_ch = out_c
                    first_layer = False

            elif name.startswith('SuperConvK1BNRELU'):
                # (in_c, out_c, stride, n)
                if len(args) != 4:
                    raise ValueError(f'{name} expects 4 args, got {len(args)}: {args}')
                in_c, out_c, stride, n = args
                if in_ch_tracker is not None and in_c != in_ch_tracker:
                    raise ValueError(f'Input channel mismatch at {name}: file says {in_c}, but current={in_ch_tracker}')
                if n != 1:
                    raise NotImplementedError(f'{name} with n={n} not supported yet.')

                features.append(
                    ConvBNReLU(in_c, out_c, kernel_size=1, stride=stride,
                               norm_layer=norm_layer, symmetric=True)
                )
                in_ch_tracker = out_c
                last_out_ch = out_c
                # 이후 first_layer는 의미 없음

            else:
                raise ValueError(f'Unknown token name: {name}')

        if last_out_ch is None:
            raise ValueError('No layers parsed from architecture.')

        self.last_channel = last_out_ch
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        ret_dict = dict()
        ret_dict['out'] = self._forward_impl(x)
        return ret_dict


def _build_from_arch_path(arch_path, num_classes=1000, norm_layer=None):
    """
    arch_path의 텍스트를 읽어 파싱 후 MobileNetFromArch로 빌드
    """
    p = Path(arch_path)
    if not p.exists():
        raise FileNotFoundError(f'Architecture file not found: {arch_path}')
    arch_str = p.read_text(encoding='utf-8')
    tokens = _parse_arch_string(arch_str)
    model = MobileNetFromArch(tokens, num_classes=num_classes, norm_layer=norm_layer)
    return model


def mobilenet_ste(pretrained=False, progress=True, arch_path=None, **kwargs):
    """
    사용법:
      - arch_path=None  => 기본 MobileNetV2 (DSQ 변형 블록) 사용
      - arch_path=...   => best_structure.txt 같은 구조 파일로부터 모델 구성
    kwargs 예: num_classes=1000, width_mult=1.0, ...
    """
    num_classes = kwargs.get('num_classes', 1000)
    norm_layer = kwargs.get('norm_layer', nn.BatchNorm2d)

    if arch_path is not None:
        # 파일 기반 구성
        model = _build_from_arch_path(arch_path, num_classes=num_classes, norm_layer=norm_layer)
        return model

    # 기본 경로
    model = MobileNetV2(**kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], progress=progress)
    #     model.load_state_dict(state_dict)
    return model
