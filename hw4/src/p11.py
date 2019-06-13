import numpy as np
import matplotlib.pyplot as plt
from utils import KNN, RBFNN


def read_data(fn):
    X = []
    y = []
    with open(fn, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split(' ')
            x = [float(x_i) for x_i in line[:-1]]
            X.append(x)
            y.append(float(line[-1]))
    return np.array(X), np.array(y)

def p11():
    train_X, train_y = read_data('hw4_train.dat')
    test_X, test_y = read_data('hw4_test.dat')
    x = [1, 3, 5, 7, 9]
    eins = []
    eouts = []
    for i in x:
        model = KNN(i)
        model.fit(train_X, train_y)

        ein = model.eval(train_X, train_y)
        eins.append(ein)
        eout = model.eval(test_X, test_y)
        eouts.append(eout)
    print(eins)
    print(eouts)
    plt.plot(x, eins)
    plt.ylim(0, 0.5)
    plt.xlabel('k')
    plt.savefig('p11.png')
    plt.clf()
    plt.plot(x, eouts)
    plt.ylim(0, 0.5)
    plt.xlabel('k')
    plt.savefig('p12.png')
    plt.clf()

def p13():
    train_X, train_y = read_data('hw4_train.dat')
    test_X, test_y = read_data('hw4_test.dat')
    gammas = [0.001, 0.1, 1, 10, 100]
    eins = []
    eouts = []
    for i in gammas:
        model = RBFNN(i)
        model.fit(train_X, train_y)

        ein = model.eval(train_X, train_y)
        eins.append(ein)
        eout = model.eval(test_X, test_y)
        eouts.append(eout)
    print(eins)
    print(eouts)
    plt.plot(np.log10(gammas), eins)
    plt.ylim(0, 0.5)
    plt.xlabel('log10(r)')
    plt.savefig('p13.png')
    plt.clf()
    plt.plot(np.log10(gammas), eouts)
    plt.ylim(0, 0.5)
    plt.xlabel('log10(r)')
    plt.savefig('p14.png')
    plt.clf()

p11()
# p13()





