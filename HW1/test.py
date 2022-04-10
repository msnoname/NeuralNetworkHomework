import pickle
from model import TwoLayerNet
from read_data import load_images, load_labels
import matplotlib.pyplot as plt


if __name__ == "__main__":
    x_test = load_images('data/t10k-images-idx3-ubyte')
    y_test = load_labels('data/t10k-labels-idx1-ubyte')

    network = pickle.load(open("hw1_model.dat", "rb"))
    test_acc = network.accuracy(x_test, y_test)
    print(str(round(test_acc*100, 3)) + "%")

    w1 = network.params['w1']
    w2 = network.params['w2']

    fig1 = plt.figure(figsize=(6, 6))
    plt.axis('off')
    for plt_index in range(100):
        ax = fig1.add_subplot(10, 10, plt_index + 1)
        ax.axis('off')
        ax.imshow(w1[:, plt_index].reshape(28, 28), cmap="gray")
    plt.show()

    fig2 = plt.figure(figsize=(2.5, 1))
    plt.axis('off')
    for plt_index in range(10):
        ax = fig2.add_subplot(2, 5, plt_index + 1)
        ax.axis('off')
        ax.imshow(w2[:, plt_index].reshape(10, 10), cmap="gray")
    plt.show()
