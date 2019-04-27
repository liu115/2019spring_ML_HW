from sklearn import linear_model
import numpy as np


def read_data(fn):
    X = []
    y = []
    with open(fn, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split(' ')
            x = [float(x_i) for x_i in line[:-1]]
            x.append(1.0) # Add x_0 = 1
            X.append(x)
            y.append(float(line[-1]))
    return np.array(X), np.array(y)

def zero_one_error(pred_y, label_y):
    assert pred_y.shape == label_y.shape
    yy = np.sign(pred_y * label_y)

    return np.sum((yy < 0).astype(np.int)) * 1. / pred_y.shape[0]


# def zero_one_error(predict_y, label_y):
#     error = 0.
#     for i, y in enumerate(label_y):
#         if predict_y[i] * label_y[i] <= 0:
#             error += 1
#
#     return error

class Ridge(object):

    def __init__(self, X, y, alpha):
        X_t = np.transpose(X)
        XX = np.matmul(X_t, X)
        self.alpha = alpha
        self.w = np.linalg.inv(XX + alpha * np.eye(X.shape[1]))

        self.w = np.matmul(self.w, X_t)
        self.w = np.matmul(self.w, y)

        assert len(self.w.shape) == 1
        assert self.w.shape[0] == X.shape[1]

    def predict(self, X):
        pred_y = np.matmul(X, self.w)

        return pred_y

    def score(self, X, y):

        pred_y = self.predict(X)
        pred_y = np.sign(pred_y)
        return zero_one_error(pred_y, y)

    def __repr__(self):

        return 'Ridge with alpha {} and w {}'.format(self.alpha, self.w)

class BagRidge(object):

    def __init__(self, X, y, alpha, sample_size, max_iter):
        np.random.seed(1126)
        self.g_list = []
        self.max_iter = max_iter

        for i in range(self.max_iter):
            sample_idx = np.random.randint(0, X.shape[0], size=sample_size)

            sample_X = X[sample_idx, :]
            sample_y = y[sample_idx]
            g = Ridge(sample_X, sample_y, alpha=alpha)
            # print(g)
            self.g_list.append(g)

    def predict(self, X):

        pred_y = np.zeros((X.shape[0]))
        for i, g in enumerate(self.g_list):
            g_pred_y = g.predict(X)
            pred_y += np.sign(g_pred_y)
        # Break tie
        pred_y[pred_y == 0] = 1
        return np.sign(pred_y)

    def score(self, X, y):
        pred_y = self.predict(X)
        return zero_one_error(pred_y, y)

ALPHAS = [.05, .5, 5., 50., 500.]
val_split = 400
def p9():
    X, y = read_data('data/hw2_lssvm_all.dat')
    train_X = X[:val_split]
    train_y = y[:val_split]
    val_X = X[val_split:]
    val_y = y[val_split:]

    eins = []
    eouts = []

    for alpha in ALPHAS:
        reg = Ridge(train_X, train_y, alpha=alpha)
        # reg.fit(train_X, train_y)

        ein = reg.score(train_X, train_y)
        eout = reg.score(val_X, val_y)
        eins.append(ein)
        eouts.append(eout)

    print('All E_in', eins)
    print('All E_out', eouts)
    print(np.argmin(eins), ALPHAS[np.argmin(eins)], np.min(eins))
    print(np.argmin(eouts), ALPHAS[np.argmin(eouts)], np.min(eouts))

def p11():
    X, y = read_data('data/hw2_lssvm_all.dat')
    train_X = X[:val_split]
    train_y = y[:val_split]
    val_X = X[val_split:]
    val_y = y[val_split:]

    eins = []
    eouts = []

    sample_size = 400
    max_iter = 250

    for alpha in ALPHAS:
        G = BagRidge(train_X, train_y, alpha, sample_size, max_iter)

        ein = G.score(train_X, train_y)
        eout = G.score(val_X, val_y)
        eins.append(ein)
        eouts.append(eout)

    print('All E_in', eins)
    print('All E_out', eouts)
    print(np.argmin(eins), ALPHAS[np.argmin(eins)], np.min(eins))
    print(np.argmin(eouts), ALPHAS[np.argmin(eouts)], np.min(eouts))


if __name__ == '__main__':
    print('############ P9')
    p9()
    print('############ P11')
    p11()
