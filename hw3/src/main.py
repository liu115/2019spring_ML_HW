import numpy as np
import matplotlib.pyplot as plt

from utils import DecisionTree, BagTree, visualize_desision_tree, zero_one_error

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
    train_X, train_y = read_data('hw3_train.dat')
    model = DecisionTree()
    model.fit(train_X, train_y)
    visualize_desision_tree(model)

def p12():
    train_X, train_y = read_data('hw3_train.dat')
    test_X, test_y = read_data('hw3_test.dat')
    model = DecisionTree()
    model.fit(train_X, train_y)

    print('E_in', model.eval(train_X, train_y))
    print('E_out', model.eval(test_X, test_y))

def p13():
    train_X, train_y = read_data('hw3_train.dat')
    test_X, test_y = read_data('hw3_test.dat')
    eins = []
    eouts = []
    for i in range(1, 7):
        model = DecisionTree(max_height=i)
        model.fit(train_X, train_y)
        eins.append(model.eval(train_X, train_y))
        eouts.append(model.eval(test_X, test_y))
    print(eins)
    print(eouts)
    x = list(range(1, 7))
    plt.plot(x, eins, label='E_in')
    plt.plot(x, eouts, label='E_out')
    plt.xlabel('H')
    plt.legend()
    plt.savefig('p13.png')



def p14_16():
    T = 30000
    train_X, train_y = read_data('hw3_train.dat')
    test_X, test_y = read_data('hw3_test.dat')
    model = BagTree(bag_size=T)
    model.fit(train_X, train_y)
    print(len(model.models))
    print(model.eval(train_X, train_y))
    print(model.eval(test_X, test_y))

    x = list(range(T))
    # 14
    ein_gt = []
    for i in range(T):
        ein_gt.append(model.models[i].eval(train_X, train_y))
    plt.hist(ein_gt)
    plt.savefig('p14.png')
    plt.clf()

    # 15, 16
    n = train_y.shape[0]
    ein_saved = np.array([[0. for i in range(n)] for j in range(T)])
    n = test_y.shape[0]
    eout_saved = np.array([[0. for i in range(n)] for j in range(T)])
    for i in range(T):
        ein_saved[i:, :] += model.models[i].predict(train_X).astype(float)
        eout_saved[i:, :] += model.models[i].predict(test_X).astype(float)
    
    # ein_saved = ein_saved.astype(int)
    ein_saved[ein_saved >= 0] = 1
    ein_saved[ein_saved < 0] = -1

    # eout_saved = eout_saved.astype(int)
    eout_saved[eout_saved >= 0] = 1
    eout_saved[eout_saved < 0] = -1

    ein_Gt = []
    eout_Gt = []

    for i in range(T):
        ein_Gt.append(zero_one_error(ein_saved[i, :], train_y))
        eout_Gt.append(zero_one_error(eout_saved[i, :], test_y))
    plt.plot(x, ein_Gt)
    plt.savefig('p15.png')
    plt.clf()
    plt.plot(x, eout_Gt)
    plt.savefig('p16.png')
    plt.clf()

if __name__ == '__main__':
    p12()
    # p14_16()
