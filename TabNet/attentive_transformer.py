# ATTENTIVE TRANSFORMER

import torch
import torch.nn
from torch.nn import Linear
from GBN import ghost_BN
from sparsemax import Sparsemax


class attentive_transformer(torch.nn.Module):

    def __init__(self, inp_dim, out_dim, virtual_batch_size=128, momentum=0.02, epsilon=0.1):
        super(attentive_transformer, self).__init__()

        self.fc = Linear(inp_dim, out_dim, bias = False)
        self.bn = ghost_BN(out_dim, virtual_batch_size, momentum, epsilon)
        self.out_act = Sparsemax(dim=-1)
        
    def forward(self, x, prior):
        x = self.fc(x)
        x = self.bn(x)
        x = torch.mul(x,prior)
        x = self.out_act(x)
        return x