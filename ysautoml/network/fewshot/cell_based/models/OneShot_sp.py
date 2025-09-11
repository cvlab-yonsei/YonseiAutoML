##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##############################################################################
# Random Search and Reproducibility for Neural Architecture Search, UAI 2019 # 
##############################################################################
import torch, random
import torch.nn as nn
from copy import deepcopy
from models.search_cells_op_level import ResNetBasicblock_splitable as ResNetBasicblock
from models.search_cells_op_level import NAS201SearchCell_splitable as SearchCell
from utils.genotypes        import Structure


class UniformRandomSupernet_splitable(nn.Module):
    def __init__(self, 
                 C, N, max_nodes, num_classes, 
                 search_space, 
                 affine, track_running_stats,
                 split_info, option):
        super().__init__()
        
        self._C        = C
        self._layerN   = N
        self.max_nodes = max_nodes
        self.split_info = split_info
        self.option = option

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C, affine=affine, track_running_stats=track_running_stats)
        )
        
        layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if index in self.split_info:
                split_op_list = self.split_info[index]
            else:
                split_op_list = None

            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, affine, track_running_stats,)# split_op_list=split_op_list)
            else:
                cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats, split_op_list=split_op_list)
                if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
                else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
            
            self.cells.append(cell)
            C_prev = cell.out_dim
        
        self.lastact = nn.Sequential(
            nn.BatchNorm2d(C_prev, affine=affine, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        
        self.op_names   = deepcopy(search_space)
        self._Layer     = len(self.cells)

        self.edge2index = {}
        self.edge2index['1<-0'] = 0
        self.edge2index['2<-0'] = 1
        self.edge2index['2<-1'] = 2
        self.edge2index['3<-0'] = 3
        self.edge2index['3<-1'] = 4
        self.edge2index['3<-2'] = 5


    def get_param_groups(self):
        groups = ([], [])
        for name, p in self.named_parameters():
            if 'cells' in name:
                w_str = name.split('.')
                cell_ind = int(w_str[1])
                if cell_ind in self.split_info:
                    edge_str = w_str[3]
                    curr_op_ind = int(w_str[4])
                    if edge_str in self.split_info[cell_ind]:
                        for op_name in self.split_info[cell_ind][edge_str]:
                            op_ind = self.op_names.index(op_name)
                            if curr_op_ind == op_ind or curr_op_ind == op_ind+2: # NOTE: HARD CODED (+2 for 3x3, +3 for 1x1)
                                if p.requires_grad: groups[0].append(p) 
                            else:
                                if p.requires_grad: groups[1].append(p) 
                    else:
                        if p.requires_grad: groups[1].append(p) 
                else:
                    if p.requires_grad: groups[1].append(p) 
                    
            else:
                if p.requires_grad: groups[1].append(p) 
        return groups


    def freeze_part(self):
        for name, p in self.named_parameters():
            if 'cells' in name:
                w_str = name.split('.')
                cell_ind = int(w_str[1])
                if cell_ind in self.split_info:
                    edge_str = w_str[3]
                    curr_op_ind = int(w_str[4])
                    if edge_str in self.split_info[cell_ind]:
                        for op_name in self.split_info[cell_ind][edge_str]:
                            op_ind = self.op_names.index(op_name)
                            if curr_op_ind == op_ind or curr_op_ind == op_ind+2: # NOTE: HARD CODED (+2 for 3x3, +3 for 1x1)
                                p.requires_grad = True
                            else:
                                p.requires_grad = False
                    else:
                        p.requires_grad = False
                else:
                    p.requires_grad = False
                    
            else:
                p.requires_grad = False

    
    def check_arch(self, rand_arch):
        arch_dict = {}
        arch_dict[0] = rand_arch[0][0][0]
        arch_dict[1] = rand_arch[1][0][0]
        arch_dict[2] = rand_arch[1][1][0]
        arch_dict[3] = rand_arch[2][0][0]
        arch_dict[4] = rand_arch[2][1][0]
        arch_dict[5] = rand_arch[2][2][0]

        new_dict = {}
        for _, split_dict in self.split_info.items():
            for edge_str, op_list in split_dict.items():
                edge_ind = self.edge2index[edge_str]
                if edge_ind in new_dict:
                    new_dict[edge_ind] += op_list
                else:
                    new_dict[edge_ind] = op_list

        for k, v in new_dict.items():
            new_dict[k] = list(set(v))

    #     new_dict = dict(sorted(new_dict.items(), key=lambda item: item[0]))

        check_list = []
        for edge_ind, op_list in new_dict.items():
            if arch_dict[edge_ind] in op_list:
                check_list += [True]
            else:
                check_list += [False]

        return any(check_list)


    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return ('{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

#     def random_genotype(self):
#         genotypes = []
#         for i in range(1, self.max_nodes):
#             xlist = []
#             for j in range(i):
#                 op_name = random.choice(self.op_names)
#                 xlist.append((op_name, j))
#             genotypes.append(tuple(xlist))
#         arch = Structure(genotypes)
#         return arch

    ## NOTE: This is to avoid Trash Archs
    def random_genotype(self):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            if i==self.max_nodes-1:
                if genotypes[0][0][0]=='none':
                    if genotypes[1][0][0]=='none':
                        for j in range(i):
                            op_name = random.choice(self.op_names[1:])
                            xlist.append((op_name, j))
                    else:
                        for j in range(i):
                            if j==i-1 and xlist[0][0]=='none':
                                op_name = random.choice(self.op_names[1:])
                                xlist.append((op_name, j))
                            else:
                                op_name = random.choice(self.op_names)
                                xlist.append((op_name, j))
                else:
                    for j in range(i):
                        if j==i-1 and xlist[0][0]=='none' and xlist[1][0]=='none':
                            op_name = random.choice(self.op_names[1:])
                            xlist.append((op_name, j))
                        else:
                            op_name = random.choice(self.op_names)
                            xlist.append((op_name, j))
            else:
                for j in range(i):
                    op_name = random.choice(self.op_names)
                    xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        arch = Structure(genotypes)

        if self.option in [3,5]:
            if not self.check_arch(arch):
                return self.random_genotype()
            else:
                return arch
        else:
            return arch


    def forward(self, inputs, arch=None):
        feature = [ self.stem(inputs) ]
        for ind, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                if arch is not None:
                    feature = cell.forward_dynamic(feature, arch)
                else:
                    feature = cell.forward_dynamic(feature, self.random_genotype())
            else:
                feature = cell(feature)

        logit_list = []
        for fea in feature:
            out = self.lastact(fea)
            out = self.global_pooling(out)
            out = out.view(out.size(0), -1)
            logits = self.classifier(out)
            logit_list.append( logits )
        return logit_list
