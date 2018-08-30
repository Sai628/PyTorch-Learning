# coding=utf-8

import time

import torch as t
from torch import nn


class BasicModule(nn.Module):
    """
    封装了 nn.Module, 主要提供save和load两个方法
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型, 默认使用 "模型名字+时间" 作为文件名
        如: AlexNet_0830_10:00:00.pth
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name
