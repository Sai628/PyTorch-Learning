# coding=utf-8

import torchvision as tv
from torchvision import transforms as T


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


def gram_matrix(y):
    """
    计算 Gram 矩阵.
    输入: b * c * h * w
    返回: b * c * c
    """
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, h * w)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def get_style_data(path):
    """
    加载风格图片.
    输入: path 文件路径
    返回: 形状 1 * c * h * w, 分布 -2~2
    """
    style_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
    ])
    style_image = tv.datasets.folder.default_loader(path)
    style_tensor = style_transform(style_image)
    return style_tensor.unsqueeze(0)


def normalize_batch(batch):
    """
    批量标准化处理.
    输入: b * ch * h * w, 0~255
    返回: b * ch * h * w, -2~2
    """
    mean = batch.data.new(IMAGE_NET_MEAN).view(1, -1, 1, 1)
    std = batch.data.new(IMAGE_NET_STD).view(1, -1, 1, 1)
    mean = (mean.expand_as(batch.data))
    std = (std.expand_as(batch.data))
    return (batch / 255.0 - mean) / std
