# coding=utf-8

import time

import numpy as np
import visdom


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

    def plot(self, name, y, **kwargs):
        """
        用法: self.plot('loss', 1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(X=np.array([x]), Y=np.array([y]), win=name, opts=dict(title=name),
                      update=None if x == 0 else 'append', **kwargs)

        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        self.vis.images(img_.cpu().numpy(), win=name, opts=dict(title=name), **kwargs)

    def log(self, info, win='log_text'):
        """
        用法: self.log({'loss': 1, 'lr': 0.0001})
        """
        self.log_text += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win=win)

    def __getattr__(self, name):
        """
        自定义的 plot, image, log, plot_many 等除外,
        self.function 等价于 self.vis.function
        """
        return getattr(self.vis, name)
