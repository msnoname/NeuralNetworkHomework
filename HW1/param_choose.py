import numpy as np
import random
from read_data import load_images, load_labels
from model import TwoLayerNet
import matplotlib.pyplot as plt


def find_hidden_layer_size(x_train, y_train):
    random.seed(33)
    # mini-batch的实现
    # print(x_train.shape)  # 输出: [60000, 784]
    # print(y_train.shape)  # 输出: [10000, 10]
    train_size = x_train.shape[0]
    # 每次迭代的样本数
    batch_size = 100
    # 迭代计算次数
    iters_num = 1000

    # 查找隐藏层大小
    size_list = [10, 25, 50, 75, 100, 250, 500, 1000, 1500]
    train_acc = []
    test_acc = []
    for size in size_list:
        network = TwoLayerNet(input_size=784, hidden_size=size, output_size=10)
        learning_rate = 0.79
        # 迭代计算
        for i in range(iters_num):
            # 随机选取batch_size个样本
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            y_batch = y_train[batch_mask]
            # 计算梯度
            grads = network.gradient(x_batch, y_batch)
            # 梯度下降
            for key in ('w1', 'w2', 'b1', 'b2'):
                network.params[key] -= learning_rate * (grads[key])
        # 计算精确度
        train_acc.append(network.accuracy(x_train, y_train))
        test_acc.append(network.accuracy(x_test, y_test))

    plt.axes(xscale="log")
    plt.xlabel('hidden layer size (log scale)')
    plt.ylabel('accuracy')
    plt.plot(size_list, train_acc, label='train')
    plt.plot(size_list, test_acc, label='test')
    plt.legend()
    plt.show()


def find_learning_rate(x_train, y_train):
    random.seed(33)
    # mini-batch的实现
    # print(x_train.shape)  # 输出: [60000, 784]
    # print(y_train.shape)  # 输出: [10000, 10]
    train_size = x_train.shape[0]
    # 每次迭代的样本数
    batch_size = 100
    # 迭代计算次数
    iters_num = 100

    # 查找初始学习率
    network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    learning_rate = 10 ** (-5)
    loss = []
    learning_rate_list = []
    # 迭代计算
    for i in range(iters_num):
        # 随机选取batch_size个样本
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]
        # 计算梯度
        grads = network.gradient(x_batch, y_batch)
        # 梯度下降
        for key in ('w1', 'w2', 'b1', 'b2'):
            network.params[key] -= learning_rate * (grads[key])
        # 计算损失
        loss.append(network.loss(x_batch, y_batch))
        # 记录学习率
        learning_rate_list.append(learning_rate)
        # 更新学习率
        learning_rate *= 10 ** (1 / 20)

    plt.axes(xscale="log")
    plt.xlabel('learning rate (log scale)')
    plt.ylabel('loss')
    plt.plot(learning_rate_list, loss)
    plt.show()

    return learning_rate_list[np.argmin(loss)]


def find_lambda_regular(x_train, y_train):
    random.seed(33)
    # mini-batch的实现
    # print(x_train.shape)  # 输出: [60000, 784]
    # print(y_train.shape)  # 输出: [10000, 10]
    train_size = x_train.shape[0]
    # 每次迭代的样本数
    batch_size = 100
    # 迭代计算次数
    iters_num = 1000
    iter_per_epoch = max(train_size / batch_size, 1)

    # 查找正则化参数
    lambda_list = [0, 0.0001, 0.001, 0.01, 0.1, 1]
    train_acc = []
    test_acc = []
    # final_loss = []
    learning_rate = 0.79
    for lambda_regular in lambda_list:
        network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
        # 迭代计算
        # loss = []
        for i in range(iters_num):
            # 随机选取batch_size个样本
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            y_batch = y_train[batch_mask]
            # 计算梯度
            grads = network.gradient(x_batch, y_batch)
            # 梯度下降
            for key in ('w1', 'w2'):
                network.params[key] -= learning_rate * (grads[key] + lambda_regular * network.params[key])
            for key in ('b1', 'b2'):
                network.params[key] -= learning_rate * (grads[key])
            # 计算损失
            # loss.append(network.loss(x_train, y_train))
        # final_loss.append(loss)
        train_acc.append(network.accuracy(x_train, y_train))
        test_acc.append(network.accuracy(x_test, y_test))

    plt.axes(xscale="log")
    plt.xlabel('lambda regular (log scale)')
    plt.ylabel('accuracy')
    plt.plot(lambda_list, train_acc, label='train')
    plt.plot(lambda_list, test_acc, label='test')
    # for i in range(len(lambda_list)):
    #     plt.plot(range(iters_num), final_loss[i], label=lambda_list[i])
    plt.legend()
    plt.show()


if __name__ == '__main__':

    # 加载数据，分为学习数据和测试数据，x为输入，y为标签
    x_train = load_images('data/train-images-idx3-ubyte')
    y_train = load_labels('data/train-labels-idx1-ubyte')
    x_test = load_images('data/t10k-images-idx3-ubyte')
    y_test = load_labels('data/t10k-labels-idx1-ubyte')

    # find_hidden_layer_size(x_train, y_train)

    # learning_rate = find_learning_rate(x_train, y_train)
    # print(learning_rate)

    find_lambda_regular(x_train, y_train)

