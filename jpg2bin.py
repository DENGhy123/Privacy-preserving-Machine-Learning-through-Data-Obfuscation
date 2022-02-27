# -*- coding: utf-8 -*-
"""
@Time ： 2022/2/17 22:16
@Auth ： dhy
@File ：jpg2bin.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
# -*- coding:utf-8 -*-
import os
import pickle as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg
from PIL import Image
import cv2
def png2bin(gen_dir, bach_dir, label_dir):
    data_batch = {b'batch_label':b'testing batch 1 of 1', b'labels':'', b'data':'', b'filenames':''}
    data=[]
    label=[]
    filename=[]
    for b_dir in bach_dir:
        for l_dir in label_dir:
            im = os.listdir(gen_dir + '/' + b_dir + '/' + l_dir)
            for item in im:
                image_name = gen_dir + '/' + b_dir +'/' + l_dir + '/' + item
                # print(image_name)
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
                img = cv2.imread(image_name)
                for j in range(2,-1,-1):
                    for i in range(32):
                        for k in range(32):
                            data_temp.append(img[i][k][j])
                # data.append(np.array(data_temp,dtype='uint8'))
                data.append(data_temp)
                filename.append(item.encode())
    data = np.array(data, dtype='uint8')
    data_batch[b'labels'] = label
    data_batch[b'data'] = data
    data_batch[b'filenames'] = filename
    return data_batch

def save_obj(save_dir, data_batch):
    with open(save_dir +'/'+ 'test_batch', 'wb') as f:
        p.dump(data_batch, f, p.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return p.load(f)

if __name__ == "__main__":
    demo = r'F:\AI_work\DP+AI\cifar-10-batches-py\train\batch_1\frog\leptodactylus_pentadactylus_s_000004.png'
    gen_dir = r'F:\AI_work\DP+AI\cifar-10-batches-py\test'
    bach_dir = [r'train\batch_1',]
    label_dir = ['bird', 'car', 'cat', 'deer', 'dog', 'frog', 'horse', 'plane', 'ship', 'truck']
    data_batch = png2bin(gen_dir,bach_dir,label_dir)
    save_dir = r'F:\AI_work\DP+AI\data_1\cifar-10-python'
    save_obj(save_dir, data_batch)
