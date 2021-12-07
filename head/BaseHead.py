import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

class BaseHead(Module):
    def __init__(self, feat_dim, num_class):
        super(BaseHead, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, feats):
        kernel_norm = F.normalize(self.weight, dim=0)
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm) 
        cos_theta = cos_theta.clamp(-1, 1)
        return cos_theta
        