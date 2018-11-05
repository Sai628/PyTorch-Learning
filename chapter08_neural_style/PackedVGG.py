# coding=utf-8

from collections import namedtuple

import torch.nn as nn
from torchvision.models import vgg16



class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__();
        # features的第3, 8, 15, 22层分别是: relu1_2, relu2_2, relu3_3, relu4_3
        # 我们只取vgg16的前22层即可
        features = list(vgg16(pretrained=True).features)[:23]
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for i, model in enumerate(self.features):
            x = model(x)
            if i in [3, 8, 15, 22]:
                results.append(x)

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        return vgg_outputs(*results)
