import numpy as np
import matplotlib.pyplot as plt


from AdaBoostStump import DecisionStump, AdaBoost, zero_one_error


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
    print(len(X))
    return np.array(X), np.array(y)

def p13():
    max_iter = 300
    train_X, train_y = read_data('data/hw2_adaboost_train.dat')
    G = AdaBoost(max_iter=max_iter)
    G.fit(train_X[:, :], train_y[:])

    e_ins = []
    for i, g in enumerate(G.g_list):
        e_in = g.score(train_X, train_y)
        e_ins.append(e_in)

    print (G.g_list[-1].score(train_X, train_y))

    plt.plot(np.arange(max_iter), e_ins)
    plt.savefig('p13.png')

def p14():
    max_iter = 300
    train_X, train_y = read_data('data/hw2_adaboost_train.dat')
    G = AdaBoost(max_iter=max_iter)
    G.fit(train_X[:, :], train_y[:])

    e_ins = []
    for i in range(max_iter):
        e_in = G.score(train_X, train_y, first_n_g=i+1)
        e_ins.append(e_in)

    print (G.score(train_X, train_y))

    plt.plot(np.arange(max_iter), e_ins)
    plt.savefig('p14.png')

def p15():
    max_iter = 300
    train_X, train_y = read_data('data/hw2_adaboost_train.dat')
    G = AdaBoost(max_iter=max_iter)
    G.fit(train_X[:, :], train_y[:])

    U_list = []
    for i in range(max_iter):
        U = np.sum(G.weight_list[i])
        U_list.append(U)

    print(U_list[-1])

    plt.plot(np.arange(max_iter), U_list)
    plt.savefig('p15.png')

def p16():
    max_iter = 300
    train_X, train_y = read_data('data/hw2_adaboost_train.dat')
    test_X, test_y = read_data('data/hw2_adaboost_test.dat')
    G = AdaBoost(max_iter=max_iter)
    G.fit(train_X[:, :], train_y[:])

    e_ins = []
    for i in range(max_iter):
        e_in = G.score(test_X, test_y, first_n_g=i+1)
        e_ins.append(e_in)

    print (e_ins[-1])

    plt.plot(np.arange(max_iter), e_ins)
    plt.savefig('p16.png')

def main():
    p16()

main()
