# coding=utf-8

import os

import torch as t
from torch.utils.data import DataLoader
from torch.autograd import Variable as V
from torch.nn import functional as F
from torchnet import meter
from tqdm import tqdm

from config import opt
from dataset.dataset import DogCat
import models
from utils.visualize import Visualizer


def train(**kwargs):
    # 根据命令行参数更新配置
    opt.parse(kwargs)
    vis = Visualizer(opt.env, use_incoming_socket=False)

    # step1: 配置模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # step2: 数据加载
    train_data = DogCat(opt.train_data_root, train=True)
    val_data = DogCat(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # step3: 目标函数与优化器
    lr = opt.lr
    criterion = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    # step4: 统计指标: 平滑处理之后的损失, 以及混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # 开始训练
    for epoch in tqdm(range(opt.max_epoch), desc='epoch', unit='epoch'):
        loss_meter.reset()
        confusion_matrix.reset()

        for i, (data, label) in tqdm(enumerate(train_dataloader), desc='train', unit='batch'):
            # 训练模型参数
            input = V(data)
            target = V(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # 更新统计指标以及可视化
            loss_meter.add(loss.data.item())
            confusion_matrix.add(score.data, target.data)

            if i % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])

                # 如果需要的话, 进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()

        model.save()

        # 计算验证集上的指标以及可视化
        val_cm, val_accuracy = val(model, val_dataloader)
        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch}, lr:{lr}, loss:{loss}, train_cm:{train_cm}, val_cm:{val_cm}"
                .format(epoch=epoch, lr=lr, loss=loss_meter.value()[0],
                        train_cm=str(confusion_matrix.value()), val_cm=str(val_cm.value())))

        # 如果损失不再下降, 则降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息
    """

    # 把模型设为验证模式
    model.eval()

    confusion_matrix = meter.ConfusionMeter(2)
    for i, data in tqdm(enumerate(dataloader), desc='val', unit='batch'):
        input, label = data
        with t.no_grad():
            val_input = V(input)
        if opt.use_gpu:
            val_input = val_input.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(), label.long())

    # 把模型恢复为训练模式
    model.train()

    cm_value = confusion_matrix.value()
    accuracy = 100.0 * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy


def test(**kwargs):
    opt.parse(kwargs)

    # 加载模型
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # 加载测试数据
    test_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # 开始测试
    results = []
    for i, (data, path) in tqdm(enumerate(test_dataloader), desc='test'):
        with t.no_grad():
            input = V(data)
        if opt.use_gpu:
            input = input.cuda()
        score = model(input)
        probability = F.softmax(score)[:, 1].data.tolist()
        batch_results = [(path_.data.item(), probability_) for path_, probability_ in zip(path, probability)]
        results += batch_results

    write_csv(results, opt.result_file)
    return results


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


def help():
    """
    打印帮助信息: python file.py help
    """

    print("""
    usage: python {0} <function> [--args=value,]
    <function> := train | test | help
    example:
        python {0} train --env='env0830' --lr=0.01
        python {0} test --test-data-root='path/to/dataset/'
        python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire
    fire.Fire()
