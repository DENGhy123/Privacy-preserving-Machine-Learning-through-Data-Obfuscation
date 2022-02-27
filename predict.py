# -*- coding: utf-8 -*-
"""
@Time ： 2022/2/27 18:33
@Auth ： dhy
@File ：predict.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
# 导入包
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

# 数据预处理
transform = transforms.Compose(
    [transforms.Resize((32, 32)), # 首先需resize成跟训练集图像一样的大小
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

im = Image.open("plane.jpg")
im = transform(im)  # [C, H, W]
im = torch.unsqueeze(im, dim=0) # 对数据增加一个新维度，因为tensor的参数是[batch, channel, height, width]

# 实例化网络，加载训练好的模型参数
net = LeNet()
net.load_state_dict(torch.load('Lenet.pth'))
# 预测
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1].data.numpy()
print(classes[int(predict)])