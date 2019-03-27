import sys
from sklearn import svm
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
print('Input file', sys.argv[1])
print('Output file', sys.argv[2])

def read_file(in_fn):
    f = open(in_fn, 'r')
    data = []
    for line in f:
        line = line.strip()
        line = line.split()

        label = int(float(line[0]))
        inten = float(line[1])
        symm = float(line[2])

        data.append((label, inten, symm))
    return data

def make_binary(data, true_label):
    y = []
    x = []
    for i, v in enumerate(data):
        y.append(1 if v[0] == true_label else -1)
        x.append(v[1:])
    
    # print(len([_y for _y in y if _y == +1]))
    # print(len([_y for _y in y if _y == -1]))
    return y, x


def make_svm_input(y, x, out_fn):
    assert len(y) == len(x)
    with open(out_fn, 'w') as f:
        for i in range(len(x)):
            v = list([str(_x) for _x in x[i]])
            f.write('{} {}\n'.format(y[i], ' '.join(v)))


def p13():

    def train(C):
        data = read_file(sys.argv[1])
        y, x = make_binary(data, 2)
        clf = svm.SVC(kernel='linear', C=C)
        clf.fit(x, y)
        print('W:', clf.coef_)
        print('|W|', norm(clf.coef_[0]))
        return norm(clf.coef_[0])

    logC = [-5, -3, -1, 1, 3]
    n = []
    for _logC in logC:
        C = 10 ** _logC
        print(C)
        n += [train(C=C)]
    plt.plot(logC, n, 'bo')
    plt.savefig('p13.png')

def p14():
    def result(C):
        data = read_file(sys.argv[1])
        y, x = make_binary(data, 4)
        clf = svm.SVC(kernel='poly', C=C, degree=2, gamma=1, coef0=1)
        clf.fit(x, y)
        e_in = 1 - clf.score(x, y)
        return e_in

    logC = [-5, -3, -1, 1, 3]
    n = []
    for _logC in logC:
        C = 10 ** _logC
        print(C)
        n += [result(C=C)]
    print(n)
    plt.plot(logC, n, 'bo')
    plt.savefig('p14.png')

def p15():
    def K(x1, x2, gamma):
        return np.exp(-gamma * norm(x1 - x2) ** 2)

    def train(C):
        gamma = 80
        data = read_file(sys.argv[1])
        y, x = make_binary(data, 0)
        x = np.array(x)
        clf = svm.SVC(kernel='rbf', C=C, gamma=gamma, decision_function_shape='ovo')
        clf.fit(x, y)
        print(clf)
        coef = np.abs(clf.dual_coef_[0, :])
        print(coef)
        # y = np.array(y)
        # coef = np.multiply(clf.dual_coef_[0, :], y[clf.support_])
        free_idx = np.logical_and(coef > 0, coef < C)
        print(free_idx.astype(int).sum())
        sv = clf.support_vectors_[free_idx, :]
        dist = clf.decision_function(sv)
        # norm(coef_)
        print(dist[:10])

        # Calc ||w|| = sqrt(\sum alpha_i * alpha_j K(x_i, x_j))
        n_sv = clf.dual_coef_.shape[1]
        w = 0
        for i in range(n_sv):
            xi = x[clf.support_[i], :]
            for j in range(n_sv):
                xj = x[clf.support_[j], :]
                w += clf.dual_coef_[0, i] * clf.dual_coef_[0, j] * K(xi, xj, gamma)
        w = np.sqrt(w)
        print(w)
        return np.abs(dist[0]) / w

    logC = [-2, -1, 0, 1, 2]
    n = []
    for _logC in logC:
        C = 10 ** _logC
        print(C)
        n += [train(C=C)]
    plt.plot(logC, n, 'bo')
    plt.savefig('p15.png')



def p16():
    train_data = read_file(sys.argv[1])
    train_y, train_x = make_binary(train_data, 0)
    test_data = read_file(sys.argv[2])
    test_y, test_x = make_binary(test_data, 0)

    # print(len(train_x))
    # print(len(train_x[0]))
    # print(len(train_y))
    def train(x, y, gamma):
        sample_idx = np.random.permutation(len(x))
        all_train_x = np.array(x)
        all_train_y = np.array(y)

        val_x = all_train_x[sample_idx[:1000], :]
        val_y = all_train_y[sample_idx[:1000]]
        train_x = all_train_x[sample_idx[1000:], :]
        train_y = all_train_y[sample_idx[1000:]]

        clf = svm.SVC(kernel='rbf', C=0.1, gamma=gamma)
        clf.fit(train_x, train_y)

        return clf.score(val_x, val_y)
    # acc = train(train_x, train_y, 0.01)
    # print(acc)

    max_id = []
    for i in range(100):
        
        acc = []
        log_gammas = [-2, -1, 0, 1, 2]
        for log_gamma in log_gammas:
            gamma = 10 ** log_gamma
            acc += [train(train_x, train_y, gamma)]
        acc = np.array(acc)
        max_id += [ log_gammas[np.argmax(acc)] ]
        print(i, max_id, acc)
    
    plt.hist(max_id, np.arange(-2.9, 2.9, 0.2))
    plt.savefig('p16.png')

# plt.clf()
p13()
# plt.clf()
# p14()
# plt.clf()
# p15()
# p16()