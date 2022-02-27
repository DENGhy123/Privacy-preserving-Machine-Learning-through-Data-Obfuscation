# -*- coding: utf-8 -*-
"""
@Time ： 2022/2/27 18:23
@Auth ： dhy
@File ：train.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import time

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import hiddenlayer as hl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 手动修改存储数据的根路径root，如没有下载，download=Ture,下载CIFAR10数据，并对数据进行预处理
train_set = torchvision.datasets.CIFAR10(root='./data_0', train=True,
                                         download=False, transform=transform)
# 加载训练集，实际过程需要分批次（batch）训练
train_loader = torch.utils.data.DataLoader(train_set, batch_size=50,
                                           shuffle=True, num_workers=0)

# 10000张测试图片
test_set = torchvision.datasets.CIFAR10(root='./data_0', train=False,
                                        download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000,
                                          shuffle=False, num_workers=0)
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 获取测试集中的图像和标签，用于accuracy计算
test_data_iter = iter(test_loader)
test_image, test_label = test_data_iter.next()
#
# def imshow(img):  # 展示测试集图片和标签
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# # print labels
# print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
# # show images
# imshow(torchvision.utils.make_grid(test_label))

# 记录训练过程的指标
history = hl.History()
# 使用canvas进行可视化
canvas = hl.Canvas()

#打印网络
net = LeNet()
print('net structure:',net)
net = net.to(device)
summary(net, input_size=(3,32,32), batch_size=-1)

net.to(device) # 将网络分配到指定的device中
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

log_step_interval = 1000 # 记录的步数间隔

for epoch in range(20):
    print('epoch:', epoch)
    # running_loss = 0.0
    time_start = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs.to(device))				  # 将inputs分配到指定的device中
        loss = loss_function(outputs, labels.to(device))  # 将labels分配到指定的device中
        loss.backward()
        optimizer.step()
        # running_loss += loss.item()
        global_iter_num = epoch * len(train_loader) + step + 1# 计算当前是从训练开始时的第几步(全局迭代次数)
        if global_iter_num % log_step_interval == 0:
            with torch.no_grad():
                outputs = net(test_image.to(device)) # 将test_image分配到指定的device中
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy = (predict_y == test_label.to(device)).sum().item() / test_label.size(0) # 将test_label分配到指定的device中

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, loss.item(), accuracy))

                print('%f s' % (time.perf_counter() - time_start))
                running_loss = 0.0

                # 以epoch和step为索引，创建日志字典
                history.log((epoch, step),
                        train_loss=loss,
                        test_acc=accuracy)
                        # hidden_weight=LeNet.fc3[2].weight)

                # 可视化
                with canvas:
                    canvas.draw_plot(history["train_loss"])
                    canvas.draw_plot(history["test_acc"])
                    # canvas.draw_image(history["hidden_weight"])

print('Finished Training')

#保存模型
save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)