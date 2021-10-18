# ------------------------------------------------------------
# Simple Neuron Network
# From tutorial of Victor Zhou
# ------------------------------------------------------------


import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivate_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def MSE_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


# Simple NN with 1 hidden layer (2 Neurons h1-h2), 1 output o1
# h1(w11, w12, b1) - h2(w21, w22, b2) - o1(w31, w32, b3)
class NeuralNetwork:
    def __init__(self):
        self.w11 = np.random.normal()
        self.w12 = np.random.normal()
        self.w21 = np.random.normal()
        self.w22 = np.random.normal()
        self.w31 = np.random.normal()
        self.w32 = np.random.normal()
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w11 * x[0] + self.w12 * x[1] + self.b1)
        h2 = sigmoid(self.w21 * x[0] + self.w22 * x[1] + self.b2)
        o1 = sigmoid(self.w31 * h1 + self.w32 * h2 + self.b3)
        return o1

    def train(self, data, y_trues):
        learn_rate = 0.2
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, y_trues):
                sum_h1 = self.w11 * x[0] + self.w12 * x[1] + self.b1
                h1 = sigmoid(sum_h1)
                sum_h2 = self.w21 * x[0] + self.w22 * x[1] + self.b2
                h2 = sigmoid(sum_h2)
                sum_o1 = self.w31 * h1 + self.w32 * h2 + self.b3
                o1 = sigmoid(sum_o1)

                d_L_d_o1 = 2 * (o1 - y_true)

                d_o1_d_w31 = h1 * derivate_sigmoid(sum_o1)
                d_o1_d_w32 = h2 * derivate_sigmoid(sum_o1)
                d_o1_d_b3 = derivate_sigmoid(sum_o1)

                d_o1_d_h1 = self.w31 * derivate_sigmoid(sum_o1)
                d_o1_d_h2 = self.w32 * derivate_sigmoid(sum_o1)

                d_h1_d_w11 = x[0] * derivate_sigmoid(sum_h1)
                d_h1_d_w12 = x[1] * derivate_sigmoid(sum_h1)
                d_h1_d_b1 = derivate_sigmoid(sum_h1)

                d_h2_d_w21 = x[0] * derivate_sigmoid(sum_h2)
                d_h2_d_w22 = x[1] * derivate_sigmoid(sum_h2)
                d_h2_d_b2 = derivate_sigmoid(sum_h2)

                self.w11 -= learn_rate * d_L_d_o1 * d_o1_d_h1 * d_h1_d_w11
                self.w12 -= learn_rate * d_L_d_o1 * d_o1_d_h1 * d_h1_d_w12
                self.b1 -= learn_rate * d_L_d_o1 * d_o1_d_h1 * d_h1_d_b1

                self.w21 -= learn_rate * d_L_d_o1 * d_o1_d_h2 * d_h2_d_w21
                self.w22 -= learn_rate * d_L_d_o1 * d_o1_d_h2 * d_h2_d_w22
                self.b2 -= learn_rate * d_L_d_o1 * d_o1_d_h2 * d_h2_d_b2

                self.w31 -= learn_rate * d_L_d_o1 * d_o1_d_w31
                self.w32 -= learn_rate * d_L_d_o1 * d_o1_d_w32
                self.b3 -= learn_rate * d_L_d_o1 * d_o1_d_b3

                if epoch % 10 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)
                    loss = MSE_loss(y_trues, y_preds)
                    print("Epoch %d loss: %.5f" % (epoch, loss))


data = np.array([[-2, -1], [25, 6], [17, 4], [-15, -6]])
y_trues = np.array([1, 0, 0, 1])

network = NeuralNetwork()
network.train(data, y_trues)
a = np.array([-7, -3])
b = np.array([20, 2])
print("A: %.5f" % network.feedforward(a))
print("B: %.5f" % network.feedforward(b))