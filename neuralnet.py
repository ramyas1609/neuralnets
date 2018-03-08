import sys
import numpy as np
from math import *

labels = []
pixels = []


def read_csv(filename):
    train_csv_file_ = open(filename)
    for line in train_csv_file_.readlines():
        elements = line.rstrip().split(",")
        labels.append(elements[0])
        pixels.append([1] + elements[1:])


def sigmoid_forward(a):
    return 1.0 / (1 + np.exp(-a))


def sigmoid_backward(b, gb):
    gb1 = np.delete(gb, 0)
    b1 = np.delete(b, 0)
    return np.multiply(np.multiply(gb1, b1), (1 - b1))


def softmax_forward(a):
    ea = np.exp(a)
    exp_sum = np.sum(ea)
    return ea/exp_sum


def softmax_backward(b, gb):
    t0 = np.dot(b.transpose(), b)
    t1 = np.diag(b)
    return np.dot(gb.transpose(), t1 - t0)


def linear_forward(a, w):
    return np.dot(w, a)


def linear_backward(a, w, gb):
    #gw = np.dot(gb[:, None], a[None, :])
    #ga = np.dot(w.transpose(), gb)
    gw = np.outer(gb, a)
    ga = np.dot(w.transpose(), gb)
    return gw, ga


def crossentropy_forward(a, acap):
    return - (np.log(acap)[a - 1])


def crossentropy_backward(a, acap, gb):
    t0 = gb * 1.0 / acap[a - 1]
    gacap = np.zeros(shape=i_num_classes)
    gacap[a - 1] = t0
    return gacap


def nnforward(x, y, al, be):
    print x , al
    a = linear_forward(x, al)
    print a
    z = sigmoid_forward(a)
    z1 = np.insert(z, 0, 1.0)
    b = linear_forward(z1, be)
    ycap = softmax_forward(b)
    j = crossentropy_forward(y, ycap)
    return [a, z1, b, ycap, j]


def nnbackward(x, y, o):
    a = o[0]
    z = o[1]
    b = o[2]
    ycap = o[3]
    j = o[4]
    gj = 1.0
    gycap = crossentropy_backward(y, ycap, gj)
    gb = softmax_backward(ycap, gycap)
    gbe, gz = linear_backward(z, b, gb)
    ga = sigmoid_backward(z, gz)
    gal, gx = linear_backward(x, a, ga)
    return gal, gbe


def sgd(num_epoch, hidden_units, init_mode, num_classes):
    alpha = np.zeros(shape=(hidden_units, 128 + 1), dtype=float)
    alpha[:, 0] = 1.0
    beta = np.zeros(shape=(num_classes, hidden_units + 1), dtype=float)
    beta[:, 0] = 1.0
    temp_alpha = np.zeros(shape=(hidden_units, 128 + 1), dtype=float)
    temp_alpha[:, 0] = 1.0
    temp_beta = np.zeros(shape=(num_classes, hidden_units + 1), dtype=float)
    temp_beta[:, 0] = 1.0
    for e in xrange(0, num_epoch):
        for i in xrange(0, 1):
            print i
            x = np.asarray(pixels[i], dtype=float)
            o = nnforward(x, int(labels[i]), alpha, beta)
            galpha, gbeta = nnbackward(x, int(labels[i]), o)
            alpha = alpha - (0.1 * galpha)
            beta = beta - (0.1 * gbeta)
    return alpha, beta


i_train_csv = sys.argv[1]
i_num_epoch = int(sys.argv[6])
i_hidden_units = int(sys.argv[7])
i_init_mode = int(sys.argv[8])
i_learning_rate = float(sys.argv[9])
i_num_classes = int(9)

read_csv(i_train_csv)

alpha_1, beta_1 = sgd(i_num_epoch, i_hidden_units, i_init_mode, i_num_classes)

print alpha_1,beta_1