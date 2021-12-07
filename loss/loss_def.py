import yaml
from torch.nn import CrossEntropyLoss as CELoss
from .GBCosFace import GBCosFace
from .GBMagFace import GBMagFace


class LossFactory:
    """Factory to produce loss according the loss_conf.yaml.

    Attributes:
        loss_type(str): which backbone will produce.
        loss_param(dict):  parsed params and it's value. 
    """
    def __init__(self, loss_type, loss_conf_file, local_rank=None):
        self.loss_type = loss_type
        self.local_rank = local_rank
        with open(loss_conf_file) as f:
            loss_conf = yaml.load(f)
            self.loss_param = loss_conf[loss_type]
        print('loss param:')
        print(self.loss_param)

    def get_loss(self):
        if self.loss_type == 'CELoss':
            loss = CELoss()
        elif self.loss_type == 'GBCosFace':
            loss = GBCosFace(
                local_rank=self.local_rank,
                s=self.loss_param['s'],
                min_cos_v=self.loss_param['min_cos_v'],
                max_cos_v=self.loss_param['max_cos_v'],
                margin=self.loss_param['margin'],
                update_rate=self.loss_param['update_rate'],
                alpha=self.loss_param['alpha']
                )
        elif self.loss_type == 'GBMagFace':
            loss = GBMagFace(
                local_rank=self.local_rank,
                s=self.loss_param['s'],
                min_cos_v=self.loss_param['min_cos_v'],
                max_cos_v=self.loss_param['max_cos_v'],
                update_rate=self.loss_param['update_rate'],
                alpha=self.loss_param['alpha']
                )
        else:
            pass
        return loss
