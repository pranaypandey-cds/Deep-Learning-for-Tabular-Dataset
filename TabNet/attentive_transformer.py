# ATTENTIVE TRANSFORMER

import torch
import torch.nn
from torch.nn import Linear
from GBN import ghost_BN
from sparsemax import Sparsemax


class attentive_transformer(torch.nn.Module):

    def __init__(self, inp_dim, out_dim, virtual_batch_size=128, momentum=0.02, epsilon=0.1):
        super(attentive_transformer, self).__init__()

        # Linear layer
        self.fc = Linear(inp_dim, out_dim, bias = False)
        # BN layer
        self.bn = ghost_BN(out_dim, virtual_batch_size, momentum, epsilon)
        # sparsemax
        self.out_act = Sparsemax(dim=-1)
        
    def forward(self, x, prior):
        x = self.fc(x)
        x = self.bn(x)
        # elementwise multiplication with prior
        x = torch.mul(x,prior)
        # sparsemax
        x = self.out_act(x)
        return x