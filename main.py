import os
import pickle
import sys

import numpy as np

from common.functions import sigmoid, softmax
from common.gradient import numerical_gradient

sys.path.append(os.pardir)

from dataset.mnist import load_mnist


def get_data():
    # (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


def cross_entropy_error(y, t):
    """
    CEE
    :param y: Numpy Array
    :param t: Numpy Array
    :return: CEE number (float)
    """
    delta = 1e-5  # np.log() 함수에 0을 입력하면 마이너스 무한대를 뜻하게 돼 더 이상 계산을 할 수 없기 때문이다. 아주 작은 값을 더해서 0이 되지 않도록 해주는 역할이다.
    # return -np.sum(t * np.log(y + delta))
    if y.ndim == 1:
        # 데이터가 1차원이라면, 즉 데이터 하나당 교차 엔트로피 오차를 구하는 경우는 reshape 함수로 데이터의 형상을 바꿔준다.
        # 그리고 배치의 크기로 나눠 정규화하고 이미지 1자당 평균의 교차 엔트로피 오차를 계산한다.
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size


# x, t = get_data()
# network = init_network()
#
# accuracy_count = 0
# start = datetime.now()
# for i in range(len(x)):
#     y = predict(network, x[i])
#     p = np.argmax(y)  # 확률이 가장 높은 원소의 인덱스를 얻는다.
#     if p == t[i]: accuracy_count += 1
#
# print(f'Accurary: {float(accuracy_count) / len(x)}')
# print(f'Time taken: {datetime.now() - start}')
#
# batch_size = 100
# accuracy_count = 0
# start = datetime.now()
# for i in range(0, len(x), batch_size):
#     x_batch = x[i: i + batch_size]
#     y_batch = predict(network, x_batch)
#     p = np.argmax(y_batch, axis=1)
#     accuracy_count += np.sum(p == t[i: i + batch_size])
#
# print(f'Batch Accuracy: {float(accuracy_count) / len(x)}')
# print(f'Time taken: {datetime.now() - start}')


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


# net = SimpleNet()
# print(f'Init: {net.W}')
#
# x = np.array([.6, .9])
# p = net.predict(x)
# print(f'Predict: {p}')
#
# print(f'최댓값의 인덱스: {np.argmax(p)}')
#
# t = np.array([0, 0, 1])
# print(f'정답레이블과 손실: {net.loss(x, t)}')
#

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        # dx = dout * 1
        # dy = dout * 1
        # return dx + dy
        return dout * 2


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x, self.dW, self.db = None, None, None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 손실
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블 (one-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.matmul(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')

        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []

        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * .5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout = 1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)

        da *= .5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None