# coding=utf-8

import os

import torch as t
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter
import torchvision as tv
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import ipdb

from config import opt
from utils import utils
from PackedVGG import Vgg16
from transformer_net import TransformerNet


def train(**kwargs):
    opt.parse(kwargs)

    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    if opt.vis:
        from utils.visualize import Visualizer
        vis = Visualizer(env=opt.env, use_incoming_socket=False)

    # 数据加载
    transforms = T.Compose([
        T.Resize(opt.image_size),
        T.CenterCrop(opt.image_size),
        T.ToTensor(),
        T.Lambda(lambda x: x.mul(255))  # 这里图片加载后, 输出的像素值保存为0-255
    ])
    dataset = ImageFolder(opt.data_path, transforms)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=True, num_workers=opt.num_workers)


    # 转换网络 Image Transform Net
    tranformer = TransformerNet()
    if opt.model_path:
        tranformer.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    tranformer.to(device)

    # 损失网络 Vgg16
    vgg = Vgg16().eval()  # 把Vgg网络模型设为验证模式. 因为这里的损失网络不用训练, 只是用预训练好的模型来计算知觉特征和风格特征
    vgg.to(device)
    for param in vgg.parameters():
        param.requires_grad = False

    # 优化器
    optimizer = t.optim.Adam(tranformer.parameters(), opt.lr)

    # 获取风格图片的数据
    style = utils.get_style_data(opt.style_path)
    if opt.vis:
        vis.img('style', (style.data[0] * 0.225 + 0.45).clamp(min=0, max=1))
    style = style.to(device)

    # 风格图片的 gram 矩阵
    with t.no_grad():
        features_style = vgg(style)
        gram_style = [utils.gram_matrix(y) for y in features_style]

    # 损失统计
    style_meter = AverageValueMeter()
    content_meter = AverageValueMeter()

    for epoch in tqdm(range(opt.max_epoch), desc='epoch', unit='epoch'):
        style_meter.reset()
        content_meter.reset()

        for i, (x, _) in tqdm(enumerate(dataloader), desc='train', unit='batch'):
            # 训练
            optimizer.zero_grad()
            x = x.to(device)
            y = tranformer(x)
            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)
            features_y = vgg(y)
            features_x = vgg(x)

            # content loss
            content_loss = opt.content_weight * F.mse_loss(features_y.relu3_3, features_x.relu3_3)

            # style loss
            style_loss = 0.0
            for ft_y, gm_s in zip(features_style, gram_style):
                gram_y = utils.gram_matrix(ft_y)
                style_loss += F.mse_loss(gram_y, gm_s.expand_as(gram_y))
            style_loss *= opt.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            # 损失平滑
            content_meter.add(content_loss.item())
            style_meter.add(style_loss.item())

            # 可视化
            if opt.vis and i % opt.plot_every == opt.plot_every - 1:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                vis.plot('content_loss', content_meter.value()[0])
                vis.plot('style loss', style_meter.value()[0])

                # 因为x和y经过标准化处理(utils.normalize_batch), 所以需要将它们还原
                vis.img('output', (y.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
                vis.img('input', (x.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))

        # 保存visdom和模型
        if opt.vis:
            vis.save([opt.env])
        t.save(tranformer.state_dict(), 'checkpoints/style_%s.pth' % epoch)


@t.no_grad()
def stylize(**kwargs):
    opt.parse(kwargs)

    device = t.device('cuda') if opt.use_gpu else t.device('cpu')

    # 图片处理
    content_img = tv.datasets.folder.default_loader(opt.content_path)
    content_transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x.mul(255))  # 这里图片加载后, 输出的像素值保存为0-255
    ])
    content_img = content_transform(content_img)
    content_img = content_img.unsqueeze(0).to(device).detach()

    # 模型
    style_model = TransformerNet().eval()
    style_model.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    style_model.to(device)

    # 风格迁移及保存结果图片
    output = style_model(content_img)
    output_data = output.cpu().data[0]
    tv.utils.save_image(((output_data / 255)).clamp(min=0, max=1), opt.result_path)


if __name__ == '__main__':
    import fire
    fire.Fire()
