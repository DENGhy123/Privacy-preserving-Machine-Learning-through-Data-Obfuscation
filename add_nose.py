# -*- coding: utf-8 -*-
"""
@Time ： 2022/2/19 20:51
@Auth ： dhy
@File ：add_nose.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
import os
import pickle as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg
from PIL import Image
import cv2
from tqdm import tqdm

def gasuss_noise(image, mean, var):
    '''
        添加高斯噪声
        image:原始图像
        mean : 均值
        var : 方差,越大，噪声越大
    '''
    image_0 = cv2.imread(image)
    image = image_0[0:10]
    image = np.array(image/255, dtype=float)#将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
    noise = np.random.normal(mean, var ** 0.5, image.shape)#创建一个均值为mean，方差为var呈高斯分布的图像矩阵
    out = image + noise#将噪声和原始图像进行相加得到加噪后的图像
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)#clip函数将元素的大小限制在了low_clip和1之间了，小于的用low_clip代替，大于1的用1代替
    out = np.uint8(out*255)#解除归一化，乘以255将加噪后的图像的像素值恢复
    #cv.imshow("gasuss", out)
    noise = noise*255
    out = np.vstack((out,image_0[10:32]))
    return out

def png2bin(gen_dir, b_dir, label_dir, b_l_name, mean, var):
    """
    将添加噪声后的图像转写为字典
    :param gen_dir: 根路径
    :param b_dir: 每个batch的目录
    :param label_dir: 每个类别的目录
    :param b_l_name: 每个字典的标签名称
    :param mean: 噪声均值
    :param var: 噪声的方差
    :return: 返回字典
    """
    data_batch = {'batch_label':b_l_name, 'labels':'', 'data':'', 'filenames':''}
    data=[]
    label=[]
    filename=[]
    for l_dir in label_dir:
        im = os.listdir(gen_dir + '/' + b_dir + '/' + l_dir)
        for item in im:
            image_name = gen_dir + '/' + b_dir +'/' + l_dir + '/' + item
            if 'plane' in image_name:
                label.append(int(0))
            elif 'car' in image_name:
                label.append(int(1))
            elif 'bird' in image_name:
                label.append(int(2))
            elif 'cat' in image_name:
                label.append(int(3))
            elif 'deer' in image_name:
                label.append(int(4))
            elif 'dog' in image_name:
                label.append(int(5))
            elif 'frog' in image_name:
                label.append(int(6))
            elif 'horse' in image_name:
                label.append(int(7))
            elif 'ship' in image_name:
                label.append(int(8))
            else:
                label.append(int(9))
            data_temp = []
            img = gasuss_noise(image_name, mean, var)
            for j in range(2,-1,-1):
                for i in range(32):
                    for k in range(32):
                        data_temp.append(img[i][k][j])
            # data.append(np.array(data_temp,dtype='uint8'))
            data.append(data_temp)
            filename.append(item.encode())
    data = np.array(data,dtype='uint8')
    data_batch['labels'] = label
    data_batch['data'] = data
    data_batch['filenames'] = filename
    return data_batch

def save_obj(save_dir, save_batch_dir, data_batch):
    """
    将加载噪声后的图像保存为字典
    :param save_dir: 根目录
    :param save_batch_dir: 每个字典的名称
    :param data_batch: 字典
    :return:
    """
    with open(save_dir +'/'+ save_batch_dir, 'wb') as f:
        p.dump(data_batch, f, p.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # demo = r'F:\AI_work\DP+AI\cifar-10-batches-py\train\batch_1\frog\leptodactylus_pentadactylus_s_000004.png'
    # out = gasuss_noise(demo, mean=0, var=0.0001)

    #加载数据的路径，根据自己需要修改。
    gen_dir = r'F:\AI_work\DP+AI\cifar-10-batches-py'
    batch_dir = [r'train\batch_1', r'train\batch_2', r'train\batch_3', r'train\batch_4', r'train\batch_5', r'test']
    label_dir = ['bird', 'car', 'cat', 'deer', 'dog', 'frog', 'horse', 'plane', 'ship', 'truck']
    batch_label_name = ['training batch 1 of 5', 'training batch 2 of 5', 'training batch 3 of 5',
                        'training batch 4 of 5', 'training batch 5 of 5', 'testing batch 1 of 1']

    #噪声大小，根据自己需要修改。
    mean = 0
    var=0.33

    #保存加载噪声后的路径，根据需要自行修改。
    save_gen_dir = r'F:\AI_work\DP+AI\data_g\cifar-10-batches-py'
    save_batch_dir = [r'data_batch_1', r'data_batch_2', r'data_batch_3', r'data_batch_4', r'data_batch_5', r'test_batch']

    #进行加载噪声和保存.....
    for b_dir, b_l_name, s_b_dir in tqdm(zip(batch_dir, batch_label_name, save_batch_dir), desc='adding noise...'):
        data_batch = png2bin(gen_dir, b_dir, label_dir, b_l_name, mean, var)
        save_obj(save_gen_dir, s_b_dir, data_batch)


    # im = plt.imread(demo)
    # plt.imshow(im)  #
    # # 显示图像
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    # plt.imshow(im)  #
    # 显示图像
    # plt.imshow(out)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
