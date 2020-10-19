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
from keras.models import Sequential  # 从keras.models中导入Sequential
from keras.layers import Dense, Dropout, Flatten  # 导入 Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D  # 导入 卷积层、最大池化层
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

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 设定不用GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 调用0号GPU（第一个GPU）

epochs = 23  # 设定训练轮数
batch_size = 300  # 设定batch_size批次大小
figure_show = 1

np.random.seed(seed=0)  # for reproducibility 随机数的可重复


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
x_train, y_train = mnist_dataset['x_train'], mnist_dataset['y_train']  # 训练数据
x_test, y_test = mnist_dataset['x_test'], mnist_dataset['y_test']  # 测试数据
# the data, split between train and test sets
# with np.load('mnist.npz', allow_pickle=True) as f:
#     x_train, y_train = f['x_train'], f['y_train']
#     x_test, y_test = f['x_test'], f['y_test']

print('x_train.shape: ' + str(x_train.shape))  # (60000, 28, 28)
print('y_train.shape: ' + str(y_train.shape))  # (60000,)
print('x_test.shape: ' + str(x_test.shape))  # (10000, 28, 28)
print('y_test.shape: ' + str(y_test.shape))  # (10000,)

print(K.image_data_format())
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)  # 1x28x28
else:  # 默认 channels_last
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)  # 28x28x1
    # input_shape = (None, None, 1)  # 28x28x1

################################################################################
x_train = x_train.astype('float32')  # 数据转成float32型
x_test = x_test.astype('float32')  # 数据转成float32型
print('x_train[0,:,:,:]' + str(x_train[0, :, :, :]))
if figure_show:
    zoom_in = 1.0
    norm = matplotlib.colors.Normalize(vmin=np.min(x_train) * zoom_in,
                                       vmax=np.max(x_train) * zoom_in)

    fig = plt.figure()
    fig.set_size_inches(10.80, 6.1)
    #############################
    showID = 0
    fig.add_subplot(1, 2, 1)
    plt.imshow(x_train[showID].reshape(img_rows, img_cols), cmap='gray',
               norm=norm, aspect='auto')
    plt.title('train label = ' + str(y_train[showID]))
    fig.add_subplot(1, 2, 2)
    plt.imshow(x_test[showID].reshape(img_rows, img_cols), cmap='gray',
               norm=norm, aspect='auto')
    plt.title('test label = ' + str(y_test[showID]))

x_train /= 255  # 训练图片数据归一化
x_test /= 255  # 测试图片数据归一化
print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'train samples')  # 有多少训练样本
print(x_test.shape[0], 'test samples')  # 有多少测试样本

# convert class vectors to binary class matrices
# 将分类向量 转化成 [0 1]二分类矩阵，比如标签为5，在5的地方标为1，其余地方记为0
print('y_train[0] = ' + str(y_train[0]))
y_train_ori = y_train
y_test_ori = y_test

y_train = keras.utils.to_categorical(y_train, number_classes)
y_test = keras.utils.to_categorical(y_test, number_classes)

print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print('y_train[0,:] = ' + str(y_train[0, :]))
# 下载下来的数据集被分成两部分：60000行的训练数据集（mnist.train）
# 和10000行的测试数据集（mnist.test）。这样的切分很重要，
# 在机器学习模型设计时必须有一个单独的测试数据集不用于训练而是用来评估这个模型的性能，
# 从而更加容易把设计的模型推广到其他数据集上（也即：泛化）。

################################################################################
marker_label = '*'
markersize_label = 12
if figure_show:
    zoom_in = 1.0
    norm = matplotlib.colors.Normalize(vmin=np.min(x_train) * zoom_in,
                                       vmax=np.max(x_train) * zoom_in)

    fig = plt.figure()
    fig.set_size_inches(10.80, 6.1)
    #############################
    showID = 0
    fig.add_subplot(2, 3, 1)
    plt.imshow(x_train[showID].reshape(img_rows, img_cols), cmap='gray',
               norm=norm, aspect='auto')
    plt.title('train label = ' + str(y_train_ori[showID]))
    fig.add_subplot(2, 3, 4)
    plt.plot(range(10), y_train[showID], color='red', linestyle='None',
             marker=marker_label, markersize=markersize_label)
    #############################
    showID = int(x_train.shape[0] / 2) + 1
    fig.add_subplot(2, 3, 2)
    plt.imshow(x_train[showID].reshape(img_rows, img_cols), cmap='gray',
               norm=norm, aspect='auto')
    plt.title('train label = ' + str(y_train_ori[showID]))
    fig.add_subplot(2, 3, 5)
    plt.plot(range(10), y_train[showID], color='red', linestyle='None',
             marker=marker_label, markersize=markersize_label)
    #############################
    showID = -1
    fig.add_subplot(2, 3, 3)
    plt.imshow(x_train[showID].reshape(img_rows, img_cols), cmap='gray',
               norm=norm, aspect='auto')
    plt.title('train label = ' + str(y_train_ori[showID]))
    fig.add_subplot(2, 3, 6)
    plt.plot(range(10), y_train[showID], color='red', linestyle='None',
             marker=marker_label, markersize=markersize_label)

################################################################################
################################################################################
################################################################################
# 以下为人工神经网络定义区
model = Sequential()  # 序贯模型 Linear stack of layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=input_shape))
# 32个卷积核、卷积核大小3x3，relu激活函数

model.add(Conv2D(64, (3, 3), activation='relu'))
# 64个卷积核、卷积核大小3x3，relu激活函数

model.add(MaxPooling2D(pool_size=(2, 2)))  # 最大池化层
model.add(Dropout(0.25))  # Dropout层
# 为输入数据施加Dropout。
# Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，
# Dropout层用于防止过拟合。rate：0~1的浮点数，控制需要断开的神经元的比例

model.add(Flatten())  # Flatten层
# Flatten层用来将输入“压平”，即把多维的输入一维化，
# 常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。

model.add(Dense(128, activation='relu'))
# Dense就是常用的全连接层，所实现的运算是output = activation(dot(input, kernel)+bias)。
# 其中activation是逐元素计算的激活函数，kernel是本层的权值矩阵，
# bias为偏置向量，只有当use_bias=True才会添加。

model.add(Dropout(0.5))
model.add(Dense(number_classes, activation='softmax'))

################################################################################
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# 在训练模型之前，我们需要通过compile来对学习过程进行配置。compile接收三个参数：
# 优化器optimizer：该参数可指定为已预定义的优化器名
# 损失函数loss：该参数为模型试图最小化的目标函数，它可为预定义的损失函数名，
# 如categorical_crossentropy、mse，也可以为一个损失函数。
# 指标列表metrics：对分类问题，我们一般将该列表设置为metrics=['accuracy']。

################################################################################
model_log_save_dir = './model'
if not os.path.exists(model_log_save_dir):
    os.mkdir(model_log_save_dir)
checkpointer = ModelCheckpoint(
    os.path.join(model_log_save_dir, 'model_{epoch:03d}.hdf5'), verbose=1,
    save_weights_only=False, period=1)
# 本函数将模型训练epochs轮
train_history = model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          callbacks=[checkpointer],
                          validation_data=(x_test, y_test))
# fit函数返回一个History的对象，其History.history属性记录了损失函数
# 和其他指标的数值随epoch变化的情况，
# 如果有验证集的话，也包含了验证集的这些指标变化情况

# train_history.epoch 为列表

if figure_show:
    fig = plt.figure()
    fig.set_size_inches(12.30, 7.2)
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig.add_subplot(2, 2, 1)
    plt.plot(np.array(train_history.epoch) + 1, np.log10(
        np.array(train_history.history['loss']) / np.array(
            train_history.history['loss'][0])))
    plt.ylabel('log10 (loss)')
    plt.title('训练    损失函数 loss')
    fig.add_subplot(2, 2, 2)
    plt.plot(np.array(train_history.epoch) + 1,
             np.array(train_history.history['accuracy']) * 100)
    plt.ylabel('accuracy (%)')
    plt.title('训练    预测精度 accuracy')
    fig.add_subplot(2, 2, 3)
    plt.plot(np.array(train_history.epoch) + 1,
             np.log10(
                 np.array(train_history.history['val_loss']) / np.array(
                     train_history.history['val_loss'][0])))
    plt.ylabel('log10 (val_loss)')
    plt.title('校验    损失函数 val_loss')
    fig.add_subplot(2, 2, 4)
    plt.plot(np.array(train_history.epoch) + 1,
             np.array(train_history.history['val_accuracy']) * 100)
    plt.ylabel('val_accuracy (%)')
    plt.title('校验    预测精度 val_accuracy')

model_number = findLastCheckpoint(model_log_save_dir=model_log_save_dir)
model_id = model_log_save_dir + '/model_' + format(model_number, '03d') + '.hdf5'
model = load_model(model_id, compile=True)
################################################################################
score = model.evaluate(x_test, y_test, verbose=1)
# 本函数计算在某些输入数据上模型的误差，其参数有：
# x：输入数据，与fit一样，是numpy array或numpy array的list
# y：标签，numpy array
# 本函数返回一个测试误差的标量值（如果模型没有其他评价指标），
# 或一个标量的list（如果模型还有其他的评价指标）。
print('Test loss:', score[0])
print('Test accuracy:', score[1])

################################################################################
testID = int(x_test.shape[0] / 2) + 1
y_pre = model.predict(x_test[testID].reshape(1, img_rows, img_cols, 1))
# 本函数获得输入数据对应的输出

elapsed_time = time.time() - start_time
elapsed_time_str = time2HMS(elapsed_time=elapsed_time)

################################################################################
if figure_show:
    zoom_in = 1.0
    norm = matplotlib.colors.Normalize(vmin=np.min(x_train) * zoom_in,
                                       vmax=np.max(x_train) * zoom_in)

    fig = plt.figure()
    fig.set_size_inches(10.80, 6.1)
    fig.add_subplot(1, 3, 1)
    plt.imshow(x_test[testID].reshape(img_rows, img_cols), cmap='gray',
               norm=norm, aspect='auto')
    plt.title('test label = ' + str(y_test_ori[testID]))
    ###########################################
    fig.add_subplot(1, 3, 2)
    plt.plot(range(10), y_test[testID], color='red', linestyle='None',
             marker=marker_label, markersize=markersize_label)
    plt.legend(['true'])
    ###########################################
    fig.add_subplot(1, 3, 3)
    plt.plot(range(10), y_test[testID], color='red', linestyle='None',
             marker=marker_label, markersize=markersize_label)
    plt.plot(range(10), y_pre[0], color='blue', linestyle='None',
             marker=marker_label, markersize=markersize_label)
    plt.legend(['true', 'predict'])

if figure_show:
    plt.show()
