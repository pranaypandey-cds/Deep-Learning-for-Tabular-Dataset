# GHOST Batch Normalization
import torch
import torch.nn
from torch.nn import BatchNorm1d

class ghost_BN(torch.nn.Module):

    def __init__(self, inp_dim, virtual_batch_size = 128, momentum = 0.2, epsilon = 0.1):
        super(ghost_BN, self).__init__()
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(inp_dim, momentum=momentum, eps=epsilon)

    def forward(self,x):
        # deciding chunk size
        chunk_size= int(np.ceil(x.shape[0]/self.virtual_batch_size))
        # breaking into chunks
        chunks = x.chunk(chunk_size, dim=0)
        batch_list = [self.bn(chunk) for chunk in chunks]
        return torch.cat(batch_list, dim=0)
