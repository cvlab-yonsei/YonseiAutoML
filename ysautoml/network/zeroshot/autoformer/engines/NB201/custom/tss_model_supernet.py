import sys
sys.path.append("..") # Adds higher directory to python modules path

import torch, random
import torch.nn as nn
from copy import deepcopy
from xautodl.models.cell_operations import ResNetBasicblock
from xautodl.models.cell_searchs.search_cells import NAS201SearchCell as SearchCell
from xautodl.models.cell_searchs.genotypes import Structure

def get_cell_based_tiny_net(config):
    if isinstance(config, dict):
        config = dict2config(config, None)  # to support the argument being a dict
    super_type = getattr(config, "super_type", "basic")
    group_names = ["DARTS-V1", "DARTS-V2", "GDAS", "SETN", "ENAS", "RANDOM", "generic"]
    if super_type == "basic" and config.name in group_names:
            return TinyNetworkRANDOM(
                config.C,
                config.N,
                config.max_nodes,
                config.num_classes,
                config.space,
                config.affine,
                config.track_running_stats,
            )
    else:
        NotImplementedError


class TinyNetworkRANDOM(nn.Module):
    def __init__(
        self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats
    ):
        super(TinyNetworkRANDOM, self).__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = SearchCell(
                    C_prev,
                    C_curr,
                    1,
                    max_nodes,
                    search_space,
                    affine,
                    track_running_stats,
                )
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert (
                        num_edge == cell.num_edges and edge2index == cell.edge2index
                    ), "invalid {:} vs. {:}.".format(num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch_cache = None

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self.cells), cell.extra_repr()
            )
        return string

    def extra_repr(self):
        return "{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def random_genotype(self, set_cache):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                op_name = random.choice(self.op_names)
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        arch = Structure(genotypes)
        if set_cache:
            self.arch_cache = arch
        return arch

    def extract_cell_features(self, inputs):
        cell_features = []
        
        feature = self.stem(inputs)
        if feature.requires_grad:
            feature.retain_grad()
        cell_features.append(feature)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                feature = cell.forward_dynamic(feature, self.arch_cache)
            else:
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
            if isinstance(cell, SearchCell):
                feature = cell.forward_dynamic(feature, self.arch_cache)
            else:
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
            if isinstance(cell, SearchCell):
                feature = cell.forward_dynamic(feature, self.arch_cache)
            else:
                feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits
