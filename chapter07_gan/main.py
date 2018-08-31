# coding=utf-8

import os
import torch as t
from torch.utils.data import DataLoader
from torch.autograd import Variable as V
from torchnet.meter import AverageValueMeter
import torchvision as tv
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import ipdb
from tqdm import tqdm

from model import NetG, NetD
from config import opt


def train(**kwargs):
    opt.parse(kwargs)

    if opt.vis:
        from visualize import Visualizer
        vis = Visualizer(env=opt.env, use_incoming_socket=False)

    transforms = T.Compose([
        T.Resize(opt.image_size),
        T.CenterCrop(opt.image_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder(opt.data_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                            num_workers=opt.num_workers, drop_last=True)

    # 定义网络
    netg = NetG(opt)
    netd = NetD(opt)
    map_location = lambda storage, loc: storage
    if opt.netd_path:
        netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))

    # 定义优化器与损失函数
    optimizer_g = t.optim.Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(netd.parameters(), opt.lr2, betas=(opt.beta1, 0.999))
    criterion = t.nn.BCELoss()

    # 真图片label为1, 假图片label为0
    # noises为生成网络的输入
    true_labels = V(t.ones(opt.batch_size))
    fake_labels = V(t.zeros(opt.batch_size))
    fix_noises = V(t.randn(opt.batch_size, opt.nz, 1, 1))
    noises = V(t.randn(opt.batch_size, opt.nz, 1, 1))

    error_d_meter = AverageValueMeter()
    error_g_meter = AverageValueMeter()

    if opt.gpu:
        netg.cuda()
        netd.cuda()
        criterion.cuda()
        true_labels = true_labels.cuda()
        fake_labels = fake_labels.cuda()
        fix_noises = fix_noises.cuda()
        noises = noises.cuda()

    for epoch in tqdm(range(opt.max_epoch), desc='epoch', unit='epoch'):
        for i, (img, _) in tqdm(enumerate(dataloader), desc='train', unit='batch'):
            real_img = V(img)
            if opt.gpu:
                real_img = real_img.cuda()

            # 训练判别器
            if i % opt.d_every == 0:
                optimizer_d.zero_grad()

                ## 尽可能的把真图片判别为正确
                output = netd(real_img)
                error_d_real = criterion(output, true_labels)
                error_d_real.backward()

                # 尽可能的把假图片判别为错误
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises).detach()  # 根据噪声生成假图
                output = netd(fake_img)
                error_d_fake = criterion(output, fake_labels)
                error_d_fake.backward()

                optimizer_d.step()
                error_d = error_d_real + error_d_fake
                error_d_meter.add(error_d.data.item())

            # 训练生成器
            if i % opt.g_every == 0:
                optimizer_g.zero_grad()
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises)
                output = netd(fake_img)
                error_g = criterion(output, true_labels)
                error_g.backward()
                optimizer_g.step()
                error_g_meter.add(error_g.data.item())

            # 可视化
            if opt.vis and i % opt.plot_every == opt.plot_every - 1:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                fix_fake_imgs = netg(fix_noises)
                vis.images(fix_fake_imgs.data.cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
                vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
                vis.plot('error_d', error_d_meter.value()[0])
                vis.plot('error_g', error_g_meter.value()[0])

        # 保存模型与图片
        if epoch % opt.decay_every == 0:
            tv.utils.save_image(fix_fake_imgs.data[:64], '%s/%s.png' % (opt.save_path, epoch),
                                normalize=True, range=(-1, 1))
            t.save(netd.state_dict(), 'checkpoints/netd_%s.pth' % epoch)
            t.save(netg.state_dict(), 'checkpoints/netg_%s.pth' % epoch)
            error_d_meter.reset()
            error_g_meter.reset()
            optimizer_g = t.optim.Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
            optimizer_d = t.optim.Adam(netd.parameters(), opt.lr2, betas=(opt.beta1, 0.999))


def generate(**kwargs):
    """
    随机生成动漫头像 并根据netd的分数选择较好的
    """

    opt.parse(kwargs)

    netg = NetG(opt).eval()
    netd = NetD(opt).eval()
    noises = t.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(mean=opt.gen_mean, std=opt.gen_std)
    with t.no_grad():
        noises = V(noises)

    map_location = lambda storage, loc: storage
    netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))

    if opt.gpu:
        netd.cuda()
        netg.cuda()
        noises = noises.cuda()

    # 生成图片, 并计算图片在判别器的分数
    fake_img = netg(noises)
    scores = netd(fake_img).data

    # 挑选最好的某几张
    indexes = scores.topk(opt.gen_num)[1]
    result = []
    for i in indexes:
        result.append(fake_img.data[i])

    # 保存图片
    tv.utils.save_image(t.stack(result), opt.gen_img, normalize=True, range=(-1, 1))


if __name__ == '__main__':
    import fire
    fire.Fire()
