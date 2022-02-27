# -*- coding: utf-8 -*-
"""
@Time ： 2022/2/17 21:09
@Auth ： dhy
@File ：bin2jpg.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
# -*- coding:utf-8 -*-
import pickle as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg
from PIL import Image
from tqdm import tqdm

def load_CIFAR_batch(filename):
    """
    # 读取文件
    :param filename: 字典
    :return: 图像数据，图像标签，图像名称
    """
    with open(filename, 'rb')as f:
        datadict = p.load(f,encoding='bytes')
        print(datadict)
        X = datadict[b'data']
        Y = datadict[b'labels']
        img_name = datadict[b'filenames']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y, img_name

item = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

if __name__ == "__main__":
    #有5个测试集+1个测试集，手动修改路径data_batch_1
    gen_dir = r'F:\AI_work\DP+AI\cifar-10-batches-py'
    batch_dir = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
    save_dir = [r'train\batch_1',r'train\batch_2',r'train\batch_3',r'train\batch_4',r'train\batch_5',r'test']

    for b_dir, s_dir in tqdm(zip(batch_dir, save_dir), desc='saving images...'):
        imgX, imgY, imgName = load_CIFAR_batch(gen_dir+'/'+b_dir)
        print ("正在保存图片:", b_dir)
        di = {v: k for k, v in item.items()}

        for i in range(imgX.shape[0]):
            imgs = imgX[i - 1]
            img0 = imgs[0]
            img1 = imgs[1]
            img2 = imgs[2]
            i0 = Image.fromarray(img0)
            i1 = Image.fromarray(img1)
            i2 = Image.fromarray(img2)
            img = Image.merge("RGB",(i0,i1,i2))

            pred = di[imgY[i-1]]
            # name = "img" + str(i)+"_"+str(pred)+ ".png"
            name = imgName[i-1]
            #手动修改保存的路径
            img.save(gen_dir + '/' + s_dir + '/' + pred + '/'+str(name).split('\'')[1], 'png')

    print("保存完毕.")
