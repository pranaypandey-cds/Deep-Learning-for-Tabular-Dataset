import torch
import torch.nn as nn
import numpy as np



class Embedding_Creator(nn.Module):
    def __init__(self, inp_dim, cat_col_idx, cat_col_dim, cat_emb_dim):
        super(Embedding_Creator, self).__init__()
        
        self.inp_dim = inp_dim
        self.cat_col_idx = cat_col_idx
        self.cat_col_dim = cat_col_dim
        self.cat_emb_dim = cat_emb_dim
        self.post_emd_size = self.inp_dim - len(self.cat_col_idx) + np.sum(self.cat_emb_dim)

        self.embedding_list = nn.ModuleList()

        for orig_dim, emb_dim in zip(cat_col_dim,cat_emb_dim):
            self.embedding_list.append(nn.Embedding(orig_dim, emb_dim))

        # continous (1) and category column tracker (0)
        self.col_type_tracker = torch.ones(self.inp_dim, dtype=torch.bool)
        self.col_type_tracker[self.cat_col_idx] = 0

    
    def forward(self, x):

        col = []
        cat_col_tracker = 0
        for idx,is_cont in enumerate(self.col_type_tracker):
            if is_cont:
                col.append(x[:,idx].float().view(-1,1))
            else:
                col.append(self.embedding_list[cat_col_tracker](x[:,idx]))
                cat_col_tracker +=1
    
        return torch.cat(col,dim=1)
