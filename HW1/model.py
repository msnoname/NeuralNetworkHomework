import random
import numpy as np
import pickle
from read_data import load_images, load_labels
import matplotlib.pyplot as plt


def relu(x):
    return (x + np.abs(x)) / 2.0


def relu_grad(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)  # 防止溢出
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, init_std=0.01):
        self.params = {}
        self.params['w1'] = init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params['w2'] = init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    # 预测计算函数
    def predict(self, x):
        # y = softmax(w2 * (activation_func(w1 * x + b1)) + b2)
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, w1) + b1
        z1 = relu(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)
        return y

    # 使用均方误差
    def loss(self, x, t):
        y = self.predict(x)
        return mean_squared_error(y, t)

    # 计算精度
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 梯度计算
    def gradient(self, x, t):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        batch_size = x.shape[0]
        # 先保存前向计算值
        a1 = np.dot(x, w1) + b1
        z1 = relu(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)
        # 后向计算梯度：输出层(y)->隐含层2(w2,b2)->激活函数(activation_grad)->隐含层1(w1,b1)
        # 输出层梯度
        dy = (y - t) / batch_size
        # 隐含层2的梯度
        grads['w2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        dHidden2 = np.dot(dy, w2.T)
        # 激活函数的梯度
        dRelu = relu_grad(a1) * dHidden2
        # 隐含层1的梯度
        grads['w1'] = np.dot(x.T, dRelu)
        grads['b1'] = np.sum(dRelu, axis=0)
        return grads


if __name__ == '__main__':

    random.seed(33)
    # 加载数据，分为学习数据和测试数据，x为输入，y为标签
    x_train = load_images('data/train-images-idx3-ubyte')
    y_train = load_labels('data/train-labels-idx1-ubyte')
    x_test = load_images('data/t10k-images-idx3-ubyte')
    y_test = load_labels('data/t10k-labels-idx1-ubyte')

    # mini-batch的实现
    # print(x_train.shape)  # 输出: [60000, 784]
    # print(y_train.shape)  # 输出: [10000, 10]
    train_size = x_train.shape[0]  # 60000
    # 每次迭代的样本数
    batch_size = 100
    # 迭代计算次数
    iters_num = 10000
    # 学习率
    learning_rate = 0.79
    lambda_learning_rate = 0.9
    # 正则化参数
    lambda_regular = 0
    iter_per_epoch = max(train_size / batch_size, 1)  # 600
    network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    # 迭代计算
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
        if i % 100 == 0:
            train_loss.append(network.loss(x_train, y_train))
            test_loss.append(network.loss(x_test, y_test))
            train_acc.append(network.accuracy(x_train, y_train))
            test_acc.append(network.accuracy(x_test, y_test))
        # 更新学习率
        if i % iter_per_epoch == 0:
            learning_rate *= lambda_learning_rate
            # # 计算精度并输出
            # train_acc = network.accuracy(x_train, y_train)
            # test_acc = network.accuracy(x_test, y_test)
            # print(str(round(train_acc*100, 3)) + "\t\t" + str(round(test_acc*100, 3)))

    # 保存模型
    pickle.dump(network, open("hw1_model.dat", "wb"))

    # # 最终精度
    # train_acc = network.accuracy(x_train, y_train)
    # test_acc = network.accuracy(x_test, y_test)
    # print(str(round(train_acc*100, 3)) + "\t\t" + str(round(test_acc*100, 3)))

    # # 绘制loss和accuracy曲线
    # iter = np.arange(0, 10000, 100)
    #
    # plt.xlabel('iter time')
    # plt.ylabel('accuracy')
    # plt.plot(iter, train_acc, label='train')
    # plt.plot(iter, test_acc, label='test')
    # plt.show()
    #
    # plt.xlabel('iter time')
    # plt.ylabel('loss')
    # plt.plot(iter, train_loss, label='train')
    # plt.plot(iter, test_loss, label='test')
    # plt.show()


