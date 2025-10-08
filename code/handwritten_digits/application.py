import sys
import numpy as np
import pickle
from PIL import Image
import random
import idx2numpy

# 神经网络类，保持原样
class network(object):
    def __init__(self, sizes) -> None:
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for y, x in zip(sizes[1:], sizes[:-1])]
        self.crroection_rate = 0

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epoches, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epoches):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} %".format(j+1, 100*self.evaluate(test_data)/n_test))
                self.crroection_rate = self.evaluate(test_data)/n_test
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def preprocess_mnist_img(img):
    """
    MNIST图片是0~255的灰度，需归一化并reshape为(784,1)
    """
    img = 255 - img  # MNIST是黑底白字，视训练情况可反色
    img = img / 255.0
    img = img.reshape((784, 1))
    return img

def predict_image(net, img_np):
    output = net.feedforward(img_np)
    return np.argmax(output)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python predict_mnist_label.py 图片文件 标签文件 图片索引")
        sys.exit(1)

    mnist_img_path = sys.argv[1]
    mnist_label_path = sys.argv[2]
    img_index = int(sys.argv[3])

    # 加载模型
    with open('network_model.pkl', 'rb') as f:
        net = pickle.load(f)
    print("加载网络模型成功！")

    # 读取图片和标签
    images = idx2numpy.convert_from_file(mnist_img_path)
    labels = idx2numpy.convert_from_file(mnist_label_path)
    print(f"图片集 shape: {images.shape}, 标签集 shape: {labels.shape}")

    if img_index < 0 or img_index >= images.shape[0]:
        print(f"索引超出范围，应在 0 ~ {images.shape[0]-1} 之间")
        sys.exit(1)

    img = images[img_index]
    label = labels[img_index]

    img_np = preprocess_mnist_img(img)
    digit = predict_image(net, img_np)

    print(f"MNIST图片索引 {img_index} 识别的数字是: {digit}, 真实标签是: {label}")
