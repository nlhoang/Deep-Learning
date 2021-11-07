import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist


class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters =  num_filters
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        height, width = image.shape
        for h in range(height - 2):
            for w in range(width - 2):
                im_region = image[h:(h+3), w:(w+3)]
                yield im_region, h, w

    def feedforward(self, input):
        self.last_input = input
        height, width = input.shape
        output = np.zeros((height - 2, width - 2, self.num_filters))
        for im_region, h, w in self.iterate_regions(input):
            output[h, w] = np.sum(im_region * self.filters, axis=(1, 2))
        return output

    def backpropagation(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
        self.filters -= learn_rate * d_L_d_filters
        return None  # First layer


class MaxPool2:
    def iterate_regions(self, image):
        height, width, num_filters = image.shape
        new_height = height // 2
        new_width = width // 2
        for h in range(new_height):
            for w in range(new_width):
                im_region = image[(h * 2):(h * 2 + 2), (w * 2):(w * 2 + 2)]
                yield im_region, h, w

    def feedforward(self, input):
        self.last_input = input
        height, width, num_filters = input.shape
        new_height = height // 2
        new_width = width // 2
        output = np.zeros((new_height, new_width, num_filters))
        for im_region, h, w in self.iterate_regions(input):
            output[h, w] = np.amax(im_region, axis=(0, 1))
        return output

    def backpropagation(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))
            for (i2) in range(h):
                for (j2) in range(w):
                    for (f2) in range (f):
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]
        return d_L_d_input


class Softmax:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def feedforward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        input_len, nodes = self.weights.shape
        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backpropagation(self, d_L_d_out, learn_rate):
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue

            t_exp = np.exp(self.last_totals)
            S = np.sum(t_exp)
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            d_L_d_t = gradient * d_out_d_t
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
            return d_L_d_inputs.reshape(self.last_input_shape)


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
N = 1000
train_images = train_images[:N]
train_labels = train_labels[:N]
test_images = test_images[:N]
test_labels = test_labels[:N]

conv = Conv3x3(8)
pool = MaxPool2()
softmax = Softmax(13 * 13 * 8, 10)


def feedforward(image, label):
    out = conv.feedforward((image / 255) - 0.5)
    out = pool.feedforward(out)
    out = softmax.feedforward(out)
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0
    return out, loss, acc


def backpropagation(im, label, lr=.005):
    out, loss, acc = feedforward(im, label)
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]
    gradient = softmax.backpropagation(gradient, lr)
    gradient = pool.backpropagation(gradient)
    gradient = conv.backpropagation(gradient, lr)
    return loss, acc


# Training
print('Training')
history_loss = []
history_accuracy = []
num_tests = len(test_images)
for epoch in range(3):
    print('Epoch %d' % (epoch + 1))
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i % 100 == 99:
            print('[Step %d] Past 100 steps: Average Loss: %.3f | Accuracy: %d%%' %(i + 1, loss / 100, num_correct))
            loss = 0
            num_correct = 0
        l, acc = backpropagation(im, label)
        loss += l
        num_correct += acc
    history_loss.append(loss / num_tests)
    history_accuracy.append(num_correct / num_tests)


# Testing
print('Testing')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = feedforward(im, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print('Test Loss: ', loss / num_tests)
print('Test Accuracy: ', num_correct / num_tests)

# Plt Loss, accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
fig.suptitle("MNIST classification", fontsize=14, fontweight='bold')
ax1.plot(history_loss)
ax2.plot(history_accuracy)
plt.legend()