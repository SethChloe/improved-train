from __future__ import print_function
import keras  # 导入keras深度学习框架
from keras.models import load_model

import matplotlib  # 画图用
import matplotlib.pyplot as plt  # 画图用
import numpy as np  # 数据操作用

import matplotlib.image as mpimg  # 加载图片用

import glob  # glob 文件名模式匹配，不用遍历整个目录判断每个文件是不是符合。
# glob模块用来查找文件目录和文件，glob支持*?[]这三种通配符
import re  # regular-expression 正则表达式用来处理字符串
import scipy.io as scio

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 设定不用GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 调用0号GPU（第一个GPU）

figure_show = 1

name=input('please input picture name:')
true=input('input the right number:')
iWrite = mpimg.imread(name)
true_answer = eval(true)
print(iWrite.shape)
if iWrite.shape[-1] == 3:  # 判断数据几通道，3通道RGB模式，1通道灰度模式
    r, g, b = [iWrite[:, :, i] for i in range(3)]
    iWrite = r * 0.299 + g * 0.587 + b * 0.114  # 将RGB转成灰度模式
iWrite = 1 - iWrite  # 将 白底黑字 转成 黑底白字
iWrite = iWrite.astype('float32')  # 数据转成float32型
print(iWrite.shape)

scio.savemat('./iWrite.mat', {'iWrite': np.squeeze(iWrite)})  # 保存成MATLAB文件

fig = plt.figure()
fig.set_size_inches(10.80, 6.1)
fig.add_subplot(1, 2, 1)
plt.imshow(iWrite)
fig.add_subplot(1, 2, 2)
plt.imshow(iWrite, cmap='gray')


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
number_classes = 10  # 分10类，0,1,2,3,4,5,6,7,8,9
all_classes = range(number_classes)

# input image dimensions 输入图像的大小 28x28
img_rows, img_cols = 28,28
# 每一张图片包含28像素X28像素。我们可以用一个数字数组来表示这张图片：

model_log_save_dir = './model'

x_test = iWrite.reshape(1, img_rows, img_cols, 1)
y_test = [true_answer]
y_test_ori = y_test

y_test = keras.utils.to_categorical(y_test, number_classes)

model_number = findLastCheckpoint(model_log_save_dir=model_log_save_dir)
model_id = model_log_save_dir + '/model_' + format(model_number,
                                                   '03d') + '.hdf5'
print('The test model is ' + model_id)
model = load_model(model_id, compile=True)  # 加载预训练的人工神经网络
################################################################################
################################################################################
testID = 0
y_pre = model.predict(x_test[testID].reshape(1, img_rows, img_cols, 1))
y_pre_in_number = all_classes[
    np.argmax(y_pre)]  # Return the indices of the maximum values.
print('y_pre: ' + str(y_pre))
print('The true answer is ' + str(
    true_answer) + '. And the AI prediction is ' + str(
    y_pre_in_number) + ' with probability being ' + '{0:.4f}'.format(
    np.max(y_pre) * 100) + '%.')
################################################################################
if figure_show:
    marker_label = '*'
    markersize_label = 12

    fig = plt.figure()
    fig.set_size_inches(10.80, 6.1)
    fig.add_subplot(1, 3, 1)
    plt.imshow(x_test[testID].reshape(img_rows, img_cols), cmap='gray',
               aspect='auto')
    plt.title('The true answer is ' + str(true_answer) + '.')
    ###########################################
    fig.add_subplot(1, 3, 2)
    plt.plot(range(10), y_test[testID].reshape(10, 1), color='red',
             linestyle='None',
             marker=marker_label, markersize=markersize_label)
    plt.legend(['true'])
    ###########################################
    fig.add_subplot(1, 3, 3)
    plt.plot(range(10), y_test[testID].reshape(10, 1), color='red',
             linestyle='None',
             marker=marker_label, markersize=markersize_label)
    plt.plot(range(10), y_pre[0].reshape(10, 1), color='blue', linestyle='None',
             marker=marker_label, markersize=markersize_label)
    plt.plot(range(10), y_pre[0].reshape(10, 1), color='blue', linestyle='-',
             marker=marker_label, markersize=markersize_label)
    plt.legend(['true', 'predict'])
    plt.title('The AI prediction is ' + str(
        y_pre_in_number) + ' with probability being ' + '{0:.4f}'.format(
        np.max(y_pre) * 100) + '%.')

if figure_show:
    plt.show()
