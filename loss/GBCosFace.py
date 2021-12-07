import numpy as np
import torch
import torch.cuda.comm
import torch.distributed as dist
from torch.distributed import ReduceOp, get_world_size


class GBCosFace(torch.nn.Module):
    def __init__(self, local_rank, s=32, min_cos_v=0.5, max_cos_v=0.7, margin=0.16, update_rate=0.01, alpha=0.15):
        super(GBCosFace, self).__init__()
        self.scale = s
        self.margin = margin
        self.cos_v = None
        self.min_cos_v = min_cos_v
        self.max_cos_v = max_cos_v
        self.target = torch.from_numpy(np.array([0], np.int64))
        self.update_rate = update_rate
        self.alpha = alpha
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduce=False)
        self.local_rank = local_rank

    def forward(self, cos_theta, labels, **args):
        def cal_cos_n(pi, s):
            cos_n = torch.logsumexp(s*pi, dim=-1) / s
            return cos_n.unsqueeze(-1)
        
        update_rate = self.update_rate
        batchsize = cos_theta.size()[0]
        index = torch.zeros_like(cos_theta)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index = index.bool()
        
        # pos cos similarities
        cos_p = cos_theta[index].unsqueeze(-1)
        cos_pm = cos_p - self.margin

        # neg cos similarities
        index_neg = torch.bitwise_not(index)
        cos_i = cos_theta[index_neg]
        cos_i = cos_i.view(batchsize, -1)

        # cal pv
        cos_n = cal_cos_n(cos_i, self.scale)
        cos_v = (cos_p.detach() + cos_n.detach()) / 2
        cos_v_update = torch.mean(cos_v).reshape(1)
        if self.cos_v is None:
            self.cos_v = cos_v_update
        self.cos_v = torch.clamp(self.cos_v, self.min_cos_v, self.max_cos_v)
    
        delta = self.alpha * (self.cos_v - cos_v)
        delta_for_log = torch.mean(torch.abs(delta))
        cos_v_pred = cos_v + delta
        
        # loss_p
        pos_pred = torch.cat((cos_pm, cos_v_pred), -1)
        target = self.target.expand(batchsize).to(cos_theta.device)
        pos_loss = self.cross_entropy(2 * self.scale * pos_pred, target.long())

        # loss_n
        neg_pred = torch.cat((cos_v_pred - self.margin, cos_n), -1)
        target = self.target.expand(batchsize).to(cos_theta.device)
        neg_loss = self.cross_entropy(2 * self.scale * neg_pred, target.long())
         
        # cal mean
        pos_loss = torch.mean(pos_loss) / 2
        neg_loss = torch.mean(neg_loss) / 2

        # update cos_v
        self.cos_v = (1-self.update_rate) * self.cos_v + self.update_rate * cos_v_update
        
        # Sync pv on multiple gpus
        dist.all_reduce(self.cos_v, op=ReduceOp.SUM)
        world_size = dist.get_world_size()
        self.cos_v /= world_size

        # for debug log
        delta_p = torch.softmax(2*self.scale * pos_pred.detach(), 1)
        delta_n = torch.softmax(2*self.scale * neg_pred.detach(), 1)
        delta_p = delta_p[:, 1]
        delta_n = torch.sum(delta_n[:, 1:], 1)
        delta_p_mean = torch.mean(delta_p)
        delta_n_mean = torch.mean(delta_n)

        loss = dict(loss_pos=pos_loss, loss_neg=neg_loss, cos_v=self.cos_v,delta_p=delta_p_mean, delta_n=delta_n_mean, delta=delta_for_log,update_rate=update_rate)
        return loss
