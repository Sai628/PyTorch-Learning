# coding=utf-8

import warnings


class DefaultConfig(object):
    data_path = 'data/'  # 数据集存放路径

    num_workers = 4  # 多进程加载数据所用的进程数
    image_size = 96  # 图片尺寸
    batch_size = 256
    max_epoch = 200
    lr1 = 2e-4  # 生成器的学习率
    lr2 = 2e-4  # 判别器的学习率
    beta1 = 0.5  # Adam优化器的beta1参数
    gpu = False  # 是否使用gpu
    nz = 100  # 噪声维度
    ngf = 64  # 生成器的feature map数
    ndf = 64  # 判别器的feature map数

    vis = True  # 是否使用visdom可视化
    env = 'GAN'  # visdom的env
    plot_every = 20  # 每间隔20batch, visdom画图一次

    debug_file = '/tmp/debuggan' # 存在该文件则进入debug模式
    d_every = 1  # 每1个batch训练一次判别器
    g_every = 5  # 每5个batch训练一次判别器
    decay_every = 10  # 每10个epoch保存一次模型
    netd_path = None  # 'checkpoints/netd_xxx.pth' 预训练判别器模型路径
    netg_path = None  # 'checkpoints/netg_xxx.pth'  预训练生成器模型路径

    # 只测试不训练
    save_path = 'imgs/'  # 生成图片保存路径
    gen_img = 'result.png'
    # 从512张生成的图片中保存最好的64张
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0  # 噪声的均值
    gen_std = 1  # 噪声的方差


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
