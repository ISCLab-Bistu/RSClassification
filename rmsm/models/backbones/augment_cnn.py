""" CNN for network augmentation """
import torch.nn as nn

from .augment_cells import AugmentCell
from .base_backbone import BaseBackbone
from ..builder import BACKBONES
from ..nas import ops
from ..nas.uitls import genotypes as gt

pans_best_genotype = "Genotype(normal=[[('dil_conv_5x5', 1), ('sep_conv_3x3', 0)], " \
                     "[('sep_conv_5x5', 2), ('sep_conv_3x3', 1)], [('dil_conv_3x3', 2), " \
                     "('sep_conv_5x5', 3)], [('sep_conv_3x3', 2), ('sep_conv_5x5', 0)]], " \
                     "normal_concat=range(2, 6), reduce=[[('sep_conv_3x3', 1), ('avg_pool_3x3', 0)], " \
                     "[('avg_pool_3x3', 0), ('sep_conv_3x3', 2)], [('skip_connect', 3), ('avg_pool_3x3', 2)], " \
                     "[('sep_conv_5x5', 4), ('dil_conv_3x3', 3)]], reduce_concat=range(2, 6))"

ovarian_best_genotype = "Genotype(normal=[[('dil_conv_5x5', 0), ('dil_conv_5x5', 1)], " \
                "[('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], [('dil_conv_3x3', 0), " \
                "('dil_conv_5x5', 3)], [('skip_connect', 3), ('skip_connect', 4)]], " \
                "normal_concat=range(2, 6), reduce=[[('avg_pool_3x3', 0), ('skip_connect', 1)], " \
                "[('dil_conv_3x3', 1), ('skip_connect', 0)], [('dil_conv_5x5', 3), ('sep_conv_3x3', 2)], " \
                "[('dil_conv_5x5', 4), ('dil_conv_3x3', 1)]], reduce_concat=range(2, 6))"

cell_spectrum_best_genotype = "Genotype(normal=[[('dil_conv_5x5', 0), ('sep_conv_5x5', 1)], " \
                              "[('avg_pool_3x3', 0), ('dil_conv_5x5', 1)], [('skip_connect', 2), " \
                              "('avg_pool_3x3', 0)], [('dil_conv_5x5', 1), ('avg_pool_3x3', 2)]], " \
                              "normal_concat=range(2, 6), reduce=[[('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], " \
                              "[('dil_conv_3x3', 2), ('avg_pool_3x3', 0)], [('dil_conv_3x3', 3), ('dil_conv_5x5', 2)], " \
                              "[('sep_conv_3x3', 2), ('avg_pool_3x3', 4)]], reduce_concat=range(2, 6))"

single_cell_best_Genotype = "Genotype(normal=[[('max_pool_3x3', 0), ('dil_conv_3x3', 1)], " \
                            "[('max_pool_3x3', 2), ('max_pool_3x3', 0)], [('dil_conv_5x5', 1), " \
                            "('sep_conv_3x3', 2)], [('dil_conv_3x3', 4), ('avg_pool_3x3', 0)]], " \
                            "normal_concat=range(2, 6), reduce=[[('dil_conv_5x5', 1), ('sep_conv_3x3', 0)], " \
                            "[('avg_pool_3x3', 1), ('dil_conv_3x3', 2)], [('max_pool_3x3', 3), ('max_pool_3x3', 2)], " \
                            "[('dil_conv_5x5', 4), ('sep_conv_3x3', 3)]], reduce_concat=range(2, 6))"


@BACKBONES.register_module()
class AugmentCNN(BaseBackbone):
    """ Augmented CNN model """

    def __init__(self, input_size, input_channels, init_channels, n_classes, n_layers,
                 genotype=ovarian_best_genotype,
                 stem_multiplier=3):
        """
        Args:
            input_size: size of length
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()
        self.C_in = input_channels
        self.C = init_channels
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.genotype = genotype
        self.num_classes = n_classes

        C_cur = stem_multiplier * init_channels
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm1d(C_cur)
        )

        C_pp, C_p, C_cur = C_cur, C_cur, init_channels

        self.cells = nn.ModuleList()
        self.genotype = gt.from_str(self.genotype)

        reduction_p = False
        for i in range(n_layers):
            if i in [n_layers // 3, 2 * n_layers // 3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = AugmentCell(self.genotype, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * len(cell.concat)
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(C_p, n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)

        return logits

    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p
