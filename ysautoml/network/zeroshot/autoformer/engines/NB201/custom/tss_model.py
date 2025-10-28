import sys
sys.path.append("..") # Adds higher directory to python modules path

import torch, random
import torch.nn as nn
from xautodl.models.cell_operations import ResNetBasicblock
from xautodl.models.cell_infers.cells import InferCell


# The macro structure for architectures in NAS-Bench-201
class TinyNetwork(nn.Module):
    def __init__(self, C, N, genotype, num_classes):
        super(TinyNetwork, self).__init__()
        self._C = C
        self._layerN = N

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True)
            else:
                cell = InferCell(genotype, C_prev, C_curr, 1)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self._Layer = len(self.cells)

        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self.cells), cell.extra_repr()
            )
        return string

    def extra_repr(self):
        return "{name}(C={_C}, N={_layerN}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def extract_cell_features(self, inputs):
        cell_features = []
        
        feature = self.stem(inputs)
        if feature.requires_grad:
            feature.retain_grad()
        cell_features.append(feature)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)
            
            if feature.requires_grad:
                feature.retain_grad()
            cell_features.append(feature)

        return cell_features

    def extract_cell_features_and_logits(self, inputs):
        cell_features = []
        
        feature = self.stem(inputs)
        if feature.requires_grad:
            feature.retain_grad()
        cell_features.append(feature)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)
            
            if feature.requires_grad:
                feature.retain_grad()
            cell_features.append(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return cell_features, logits

    def forward(self, inputs):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits

    def forward_pre_GAP(self, inputs):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)

        out = self.lastact(feature)
        return out