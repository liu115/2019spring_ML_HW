import numpy as np
import matplotlib.pyplot as plt
from utils import KMeans


def read_data(fn):
    X = []
    with open(fn, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split(' ')
            x = [float(x_i) for x_i in line]
            X.append(x)
    return np.array(X)

def p15():
    train_X = read_data('hw4_train.dat')
    x = [2, 4, 6, 8, 10]
    eins = []
    var_eins = []
    for k in x:
        ein = []
        for i in range(500):
            model = KMeans(k)
            model.fit(train_X)
            ein.append(model.eval())
        var_eins.append(np.var(ein))
        eins.append(np.mean(ein))
    print(eins)
    print(var_eins)

    plt.plot(x, eins)
    plt.savefig('p15.png')
    plt.clf()
    plt.plot(x, var_eins)
    plt.savefig('p16.png')
    # print(np.mean(train_X, axis=0))


p15()





