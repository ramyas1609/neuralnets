import sys
import numpy as np
from math import *

labels = []
pixels = []

validation_labels = []
validation_pixels = []

train_entropy = []
validation_entropy = []


def read_train_csv(filename):
    train_csv_file_ = open(filename)
    for line in train_csv_file_.readlines():
        elements = line.rstrip().split(",")
        labels.append(elements[0])
        pixels.append([1] + elements[1:])


def read_validation_csv(filename):
    validation_csv_file_ = open(filename)
    for line in validation_csv_file_.readlines():
        elements = line.rstrip().split(",")
        validation_labels.append(elements[0])
        validation_pixels.append([1] + elements[1:])


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
    gacap = np.zeros(shape=i_num_classes, dtype=float)
    gacap[a] = -gb / acap[a]
    return gacap


def nnforward(x, y, al, be):
    a = linear_forward(x, al)
    z = sigmoid_forward(a)
    z1 = np.insert(z, 0, 1.0)
    b = linear_forward(z1, be)
    ycap = softmax_forward(b)
    print "ycap", ycap
    j = crossentropy_forward(y, ycap)
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
    if init_mode == 2:
        alpha = np.zeros(shape=(hidden_units, 128 + 1), dtype=float)
        beta = np.zeros(shape=(num_classes, hidden_units + 1), dtype=float)
    else:
        alpha = np.random.uniform(-0.1, 0.1, size=(hidden_units, 128 + 1))
        beta = np.random.uniform(-0.1, 0.1, size=(num_classes, hidden_units + 1))

    alpha[:, 0] = 1.0
    beta[:, 0] = 1.0
    for e in xrange(0, num_epoch):
        for i in xrange(0, len(labels)):
            x = np.asarray(pixels[i], dtype=float)
            o = nnforward(x, int(labels[i]), alpha, beta)
            galpha, gbeta = nnbackward(x, int(labels[i]), beta, o)
            alpha = alpha - (i_learning_rate * galpha)
            beta = beta - (i_learning_rate * gbeta)
        train_entropy.append(calculate_crossentropy_train(alpha, beta))
        validation_entropy.append(calculate_crossentropy_validation(alpha, beta))
    return alpha, beta


def calculate_crossentropy_train(al, be):
    j = 0
    for i in xrange(0, len(labels)):
        x = np.asarray(pixels[i], dtype=float)
        o = nnforward(x, int(labels[i]), al, be)
        j = j + o[4]
    print j/len(labels)
    return j/len(labels)


def calculate_crossentropy_validation(al, be):
    j = 0
    for i in xrange(0, len(validation_labels)):
        x = np.asarray(validation_pixels[i], dtype=float)
        o = nnforward(x, int(validation_labels[i]), al, be)
        j = j + o[4]
    print j/len(validation_labels)
    return j/len(validation_labels)


def make_prediction_train(al, be):
    labels_out = open(i_train_labels, "w")
    num_error = 0
    for i in xrange(0, len(labels)):
        x = np.asarray(pixels[i], dtype=float)
        o = nnforward(x, int(labels[i]), al, be)
        max_class = np.argmax(o[3])
        if int(max_class) != int(labels[i]):
            num_error = num_error + 1
        labels_out.write(str(int(max_class)))
        labels_out.write("\n")
    labels_out.close()
    return num_error*1.0/len(labels)


def make_prediction_validation(al, be):
    labels_out = open(i_validation_labels, "w")
    num_error = 0
    for i in xrange(0, len(validation_labels)):
        x = np.asarray(validation_pixels[i], dtype=float)
        o = nnforward(x, int(validation_labels[i]), al, be)
        max_class = np.argmax(o[3])
        if int(max_class) != int(validation_labels[i]):
            num_error = num_error + 1
        labels_out.write(str(int(max_class)))
        labels_out.write("\n")
    labels_out.close()
    return num_error*1.0/len(validation_labels)


def write_metrics_out(filename):
    metrics_file = open(filename, "w")
    for i in xrange(0, i_num_epoch):
        metrics_file.write("epoch=" + str(i + 1) + " crossentropy(train): "+str(train_entropy[i])+"\n")
        metrics_file.write("epoch=" + str(i + 1) + " crossentropy(validation): " + str(validation_entropy[i]) + "\n")

    metrics_file.write("error(train): "+str(train_error)+"\n")
    metrics_file.write("error(validation): "+str(validation_error)+"\n")
    metrics_file.close()


i_train_csv = sys.argv[1]
i_validation_csv = sys.argv[2]
i_train_labels = sys.argv[3]
i_validation_labels = sys.argv[4]
i_metrics = sys.argv[5]
i_num_epoch = int(sys.argv[6])
i_hidden_units = int(sys.argv[7])
i_init_mode = int(sys.argv[8])
i_learning_rate = float(sys.argv[9])
i_num_classes = int(10)

read_train_csv(i_train_csv)
read_validation_csv(i_validation_csv)

alpha_1, beta_1 = sgd(i_num_epoch, i_hidden_units, i_init_mode, i_num_classes)

train_error = make_prediction_train(alpha_1, beta_1)
validation_error = make_prediction_validation(alpha_1, beta_1)

write_metrics_out(i_metrics)