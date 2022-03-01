环境：	
pytorch==1.2.0
python==3.7
cuda==10.0
运行：
1、bin2jpg.py 将数据集可视化
2、add_noise.py将数据集加噪声
3、加完噪声后手动将文件夹压缩为.tar.gz格式
4、运行train.py打印模型参数，完成训练，保存模型。

model.py用于定义网络，predict.py用于对自定义输入的图像进行预测。

在自己有数据集，并且路径正确的情况下，如果运行train.py的时候，如果报错：RuntimeError: Dataset not found or corrupted. You can use download=True to download it
去D:\Anaconda3-2021.05\envs\dhy_python3.7\Lib\site-packages\torchvision\utils.py
更改函数
def check_md5(fpath, md5, **kwargs):
    return True
    # return md5 == calculate_md5(fpath, **kwargs)

注：dhy_python3.7是你的虚拟环境的名称