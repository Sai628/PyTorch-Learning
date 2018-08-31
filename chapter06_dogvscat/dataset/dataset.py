# coding=utf-8

import os

from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class DogCat(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        """
        获取所有图片地址, 并根据训练、验证、测试划分数据
        """
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # test: data/test/1.jpg
        # train: data/train/cat.1.jpg
        if self.test:  # 测试
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:  # 训练或验证
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        # 划分训练验证集, 验证:训练=3:7
        if self.test:  # 测试
            self.imgs = imgs
        elif train:  # 训练
            self.imgs = imgs[:int(imgs_num * 0.7)]
        else:  # 验证
            self.imgs = imgs[int(imgs_num * 0.7):]

        self.transforms = transforms
        if self.transforms is None:
            # 数据转换操作, 训练、验证和测试的数据转换有所区别
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            # 测试集或验证集
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:  # 训练集
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """
        返回一张图片的数据
        对于测试集, 没有label, 返回图片id. 如1000.jpg返回1000
        """
        img_path = self.imgs[index]
        if self.test:
            label = int(img_path.split('.')[-2].split('/')[-1])
        else:
            # dog->1, cat->0
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
