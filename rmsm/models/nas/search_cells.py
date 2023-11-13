import torch
import torch.nn as nn
from .ops import FactorizedReduce, StdConv, MixedOp


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """

    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction):
        """
        Args:
            n_nodes: # of intermediate n_nodes Number of intermediate nodes
            C_pp: C_out[k-2] Number of output channels of the k-2th cell, which is connected to the input node 1
            C_p: C_out[k-1] Number of output channels of the k-1th cell, which is connected to the input node 2
            C: C_in[k] (current) This is the KTH cell with C input channels
            reduction_p: flag for whether the previous cell is reduction cell or not. Reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not. Reduction: Flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes  # By default =4, the connection status of four intermediate nodes in each cell is to be determined

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preproc0 = StdConv(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = StdConv(C_p, C, 1, 1, 0, affine=False)

        # generate dag
        self.dag = nn.ModuleList()

        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self.dag[i].append(op)

    def forward(self, s0, s1, w_dag):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]

        for edges, w_list in zip(self.dag, w_dag):
            s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))

            states.append(s_cur)
        s_out = torch.cat(states[2:], dim=1)
        return s_out
