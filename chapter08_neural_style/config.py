# coding=utf-8

import warnings


class DefaultConfig(object):
    data_path = 'data/'  # 数据集存放路径: data/coco/a.jpg
    style_path = 'style.jpg'  # 风格图片存放路径

    num_workers = 4  # 多线程加载数据所用的进程数
    image_size = 256  # 图片大小
    batch_size = 8  # batch大小
    lr = 1e-3  # 学习率
    max_epoch = 2  # 训练epoch数
    use_gpu = False  # 是否使用gpu

    vis = True  # 是否使用visdom可视化
    env = 'neural-style'  # visdom的env
    plot_every = 10  # 每10个batch可视化一次

    content_weight = 1e5  # content_loss 的权重
    style_weight = 1e10  # style_loss 的权重

    debug_file = '/tmp/debug_neuralstyle'  # 存放该文件则进入debug模式
    model_path = None  # 'checkpoints/style_xxx.pth' 预训练模型的路径
    content_path = 'input.png'  # 需要进行风格迁移的图片
    result_path = 'output.png'  # 风格迁移结果的保存路径


def parse(self, kwargs):
    """
    根据字典 kwargs 更新 config 参数
    """
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribute %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k + ":", getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
