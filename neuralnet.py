import sys
import numpy as np
from math import *
from decimal import Decimal


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
    t2 = np.diag(b) - np.outer(b, b)
    t3 = gb.transpose()
    return np.dot(t3, t2)


def linear_forward(a, w):
    return np.dot(w, a)


def linear_backward(a, w, gb):
    gw = np.outer(gb, a)
    ga = np.dot(w.transpose(), gb)
    return gw, ga


def crossentropy_forward(a, acap):
    return -log(acap[a])


def crossentropy_backward(a, acap, gb):
    gacap = np.zeros(shape=i_num_classes)
    gacap[a] = -gb / acap[a]
    return gacap


def nnforward(x, y, al, be):
    a = linear_forward(x, al)
    z = sigmoid_forward(a)
    z1 = np.insert(z, 0, 1.0)
    b = linear_forward(z1, be)
    ycap = softmax_forward(b)
    #print "ycap", ycap
    j = crossentropy_forward(y, ycap)
    #print "j", j
    return [a, z1, b, ycap, j]


def nnbackward(x, y, be, o):
    a = o[0]
    z = o[1]
    ycap = o[3]
    gj = 1.0
    gycap = crossentropy_backward(y, ycap, gj)
    gb = softmax_backward(ycap, gycap)
    gbe, gz = linear_backward(z, be, gb)
    ga = sigmoid_backward(z, gz)
    gal, gx = linear_backward(x, a, ga)
    return gal, gbe


def sgd(num_epoch, hidden_units, init_mode, num_classes):
    alpha = np.zeros(shape=(hidden_units, 128 + 1), dtype=float)
    alpha[:, 0] = 1.0
    beta = np.zeros(shape=(num_classes, hidden_units + 1), dtype=float)
    beta[:, 0] = 1.0
    for e in xrange(0, num_epoch):
        for i in xrange(0, len(labels)):
            #print "hi", i, e
            x = np.asarray(pixels[i], dtype=float)
            o = nnforward(x, int(labels[i]), alpha, beta)
            galpha, gbeta = nnbackward(x, int(labels[i]), beta, o)
            alpha = alpha - (i_learning_rate * galpha)
            beta = beta - (i_learning_rate * gbeta)
            #print i, "galpha", galpha
            #print i, "gbeta", gbeta
        calculate_crossentropy(alpha, beta)
    return alpha, beta


def calculate_crossentropy(al, be):
    j = 0
    for i in xrange(0, len(labels)):
        x = np.asarray(pixels[i], dtype=float)
        o = nnforward(x, int(labels[i]), al, be)
        j = j + o[4]
    print j/len(labels)


def fine_diff (x, y, theta):
    ep = exp(-5)
    grad = np.zeros(len(theta))
    for m in xrange(1, len(theta)):
        d = np.zeros(len(theta))
        d[m] = 1
        v = crossentropy_forward()


i_train_csv = sys.argv[1]
i_num_epoch = int(sys.argv[6])
i_hidden_units = int(sys.argv[7])
i_init_mode = int(sys.argv[8])
i_learning_rate = float(sys.argv[9])
i_num_classes = int(10)

read_csv(i_train_csv)

alpha_1, beta_1 = sgd(i_num_epoch, i_hidden_units, i_init_mode, i_num_classes)

