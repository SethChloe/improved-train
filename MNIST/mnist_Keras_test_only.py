'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
1 second per epoch on a NVIDIA TITAN RTX (24GB) GPU.
80 or 70 seconds per epoch on a intel i7 CPU.
'''
# 当我们开始学习编程的时候，第一件事往往是学习打印"Hello World"。
# 就好比编程入门有Hello World，机器学习入门有MNIST。
from __future__ import print_function
import keras  # 导入keras深度学习框架
from keras import backend as K
from keras.models import load_model
from keras import callbacks
from keras.callbacks import ModelCheckpoint

import matplotlib  # 画图用
import matplotlib.pyplot as plt  # 画图用
import numpy as np  # 数据操作用
import time  # 引入time模块

import glob  # glob 文件名模式匹配，不用遍历整个目录判断每个文件是不是符合。
# glob模块用来查找文件目录和文件，glob支持*?[]这三种通配符
import re  # regular-expression 正则表达式用来处理字符串

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 设定不用GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 调用0号GPU（第一个GPU）

figure_show = 1

# np.random.seed(seed=0)  # for reproducibility 随机数的可重复


################################################################################
# 将运行时间换算成时分秒便于识别的形式，e.g.,000h03m58s998ms
def time2HMS(elapsed_time=3600):
    elapsed_time_h = int(elapsed_time / 3600)  # 计算用了多少小时
    elapsed_time_m = int((elapsed_time - elapsed_time_h * 3600) / 60)  # 又多少分
    elapsed_time_s = int(
        elapsed_time - elapsed_time_h * 3600 - elapsed_time_m * 60)  # 又多少秒
    elapsed_time_ms = int((
                                  elapsed_time - elapsed_time_h * 3600 - elapsed_time_m * 60 - elapsed_time_s) * 1e3)
    # 又多少豪秒

    # 设定时间字符串格式
    elapsed_time_str = '{:0>3d}'.format(
        elapsed_time_h) + 'h' + '{:0>2d}'.format(
        elapsed_time_m) + 'm' + '{:0>2d}'.format(
        elapsed_time_s) + 's' + '{:0>3d}'.format(elapsed_time_ms) + 'ms'
    print('   Elapsed time ' + elapsed_time_str)  # 输出显示
    return elapsed_time_str  # 返回运行时间字符串


def findLastCheckpoint(model_log_save_dir):
    # glob.glob返回所有匹配的文件路径列表。
    # 它只有一个参数pathname，定义了文件路径匹配规则，
    # 这里可以是绝对路径，也可以是相对路径。
    file_list = glob.glob(os.path.join(model_log_save_dir, 'model_*.hdf5'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            # 正则 re.findall  的用法
            # （返回string中所有与pattern相匹配的全部字串，返回形式为数组）
            # 语法：findall(pattern, string, flags=0)
            result = re.findall(".*model_(.*).hdf5.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


################################################################################
start_time = time.time()  # 当前时间戳
################################################################################
number_classes = 10  # 分10类，0,1,2,3,4,5,6,7,8,9

# input image dimensions 输入图像的大小 28x28
img_rows, img_cols = 28, 28
# 每一张图片包含28像素X28像素。我们可以用一个数字数组来表示这张图片：

################################################################################
mnist_dataset = np.load('mnist.npz')  # 加载数据
x_test, y_test = mnist_dataset['x_test'], mnist_dataset['y_test']  # 测试数据

print('x_test.shape: ' + str(x_test.shape))  # (10000, 28, 28)
print('y_test.shape: ' + str(y_test.shape))  # (10000,)

model_log_save_dir = './model'


print(K.image_data_format())
if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
else:  # 默认 channels_last
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_test = x_test.astype('float32')  # 数据转成float32型

x_test /= 255  # 测试图片数据归一化

y_test_ori = y_test

y_test = keras.utils.to_categorical(y_test, number_classes)

model_number = findLastCheckpoint(model_log_save_dir=model_log_save_dir)
model_id = model_log_save_dir + '/model_' + format(model_number, '03d') + '.hdf5'
model = load_model(model_id, compile=True)
################################################################################
################################################################################
testID = np.random.randint(1, x_test.shape[0], [1, 1]) - 1
testID=testID.reshape(1)
print(testID.shape)
y_pre = model.predict(x_test[testID].reshape(1, img_rows, img_cols, 1))
# 本函数获得输入数据对应的输出

elapsed_time = time.time() - start_time
elapsed_time_str = time2HMS(elapsed_time=elapsed_time)

################################################################################
if figure_show:
    marker_label = '*'
    markersize_label = 12
    zoom_in = 1.0
    norm = matplotlib.colors.Normalize(vmin=np.min(x_test) * zoom_in,
                                       vmax=np.max(x_test) * zoom_in)

    fig = plt.figure()
    fig.set_size_inches(10.80, 6.1)
    fig.add_subplot(1, 3, 1)
    plt.imshow(x_test[testID].reshape(img_rows, img_cols), cmap='gray',
               norm=norm, aspect='auto')
    plt.title('test label = ' + str(y_test_ori[testID]))
    ###########################################
    fig.add_subplot(1, 3, 2)
    plt.plot(range(10), y_test[testID].reshape(10, 1), color='red', linestyle='None',
             marker=marker_label, markersize=markersize_label)
    plt.legend(['true'])
    ###########################################
    fig.add_subplot(1, 3, 3)
    plt.plot(range(10), y_test[testID].reshape(10, 1), color='red', linestyle='None',
             marker=marker_label, markersize=markersize_label)
    plt.plot(range(10), y_pre[0].reshape(10, 1), color='blue', linestyle='None',
             marker=marker_label, markersize=markersize_label)
    plt.legend(['true', 'predict'])

if figure_show:
    plt.show()
