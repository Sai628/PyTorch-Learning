# coding=utf-8

import time

import visdom
import torchvision as tv
import numpy as np


class Visualizer(object):
    """
    封装了 visdom 的基本操作 但是你仍然可能通过 `self.vis.function`
    或者 `self.function` 调用原生的 visdom 接口.
    比如:
        self.text('hello visdom')
        self.histogram(t.randn(1000))
        self.line(t.arange(0, 10), t.arange(1, 11))
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 画的第几个数, 相当于横坐标
        # 保存 ('loss', 23) 即 loss 的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        修改 visdom 的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y):
        """
        用法: self.plot('loss', 1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(X=np.array([x]), Y=np.array([y]), win=name, opts=dict(title=name),
                      update=None if x == 0 else 'append')

        self.index[name] = x + 1

    def img(self, name, img_):
        if len(img_.size()) < 3:
            img_ = img_.cpu().unsqueeze(0)
        self.vis.image(img_.cpu(), win=name, opts=dict(title=name))

    def img_gird_many(self, d):
        for k, v in d.items():
            self.img_gird(k, v)

    def img_gird(self, name, input_3d):
        """
        一个batch的图片转成一个网格图，i.e. input（36，64，64）
        会变成 6*6 的网格图，每个格子大小64*64
        """
        self.img(name, tv.utils.make_grid(input_3d.cpu()[0].unsqueeze(1).clamp(max=1, min=0)))

    def log(self, info, win='log_text'):
        """
        用法: self.log({'loss': 1, 'lr': 0.0001})
        """
        self.log_text += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win=win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
