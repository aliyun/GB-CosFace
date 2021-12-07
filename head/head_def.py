import sys
import yaml
sys.path.append('../../')
from head.BaseHead import BaseHead
from head.MagHead import MagHead


class HeadFactory:
    """Factory to produce head according to the head_conf.yaml
    
    Attributes:
        head_type(str): which head will be produce.
        head_param(dict): parsed params and it's value.
    """
    def __init__(self, head_type, head_conf_file, **args):
        self.head_type = head_type
        with open(head_conf_file) as f:
            head_conf = yaml.load(f)
            self.head_param = head_conf[head_type]
        if 'num_class' in args:
            self.head_param['num_class'] = args['num_class']
        print('head param:')
        print(self.head_param)
    def get_head(self):
        if self.head_type == 'BaseHead':
            feat_dim = self.head_param['feat_dim'] # dimension of the output features, e.g. 512 
            num_class = self.head_param['num_class'] # number of classes in the training set.
            head = BaseHead(feat_dim, num_class)
        elif self.head_type == 'MagHead':
            feat_dim = self.head_param['feat_dim'] # dimension of the output features, e.g. 512 
            num_class = self.head_param['num_class'] # number
            l_a = self.head_param['l_a']
            u_a = self.head_param['u_a']
            l_margin = self.head_param['l_margin']
            u_margin = self.head_param['u_margin']
            lamda = self.head_param['lamda']
            head = MagHead(feat_dim, num_class, l_a, u_a, l_margin, u_margin, lamda)
        else:
            pass
        return head
