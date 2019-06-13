import numpy as np
INF = 1e8
def zero_one_error(pred_y, label_y):
    assert pred_y.shape == label_y.shape
    yy = np.sign(pred_y * label_y)

    return np.sum((yy < 0).astype(np.int)) * 1. / pred_y.shape[0]

def dist(x1, x2):
    return np.linalg.norm(x1 - x2)


class KNN(object):

    def __init__(self, K):
        self.K = K


    def fit(self, X, y):
        self.X = X
        self.y = y

    def _predict_one(self, x):
        dists = []
        for i in range(self.X.shape[0]):
            d = dist(x, self.X[i, :])
            dists.append((d, self.y[i]))

        dists.sort(key=lambda x: x[0])
        result = sum([dists[i][1] for i in range(self.K)])
        return 1. if result >= 0 else -1.

    def predict(self, X):
        pred_y = np.empty((X.shape[0]))

        for i in range(X.shape[0]):
            pred_y[i] = self._predict_one(X[i, :])

        return pred_y

    def eval(self, X, y):

        pred_y = self.predict(X)
        return zero_one_error(pred_y, y)

class RBFNN(KNN):

    def __init__(self, gamma):
        self.gamma = gamma

    def _predict_one(self, x):
        result = 0
        for i in range(self.X.shape[0]):
            result += self.y[i] * np.exp(-self.gamma * (dist(x, self.X[i]) ** 2))
        return result

class KMeans(object):

    def __init__(self, k):
        self.k = k

    def _update_group(self):
        update_cnt = 0
        for i in range(self.X.shape[0]):
            min_dist = INF
            x = self.X[i, :]
            old_cts_id = self.cts_id[i]
            for j in range(self.cts.shape[0]):
                c = self.cts[j, :]
                d = dist(x, c)
                if d < min_dist:
                    self.cts_id[i] = j
                    min_dist = d
            if old_cts_id != self.cts_id[i]:
                update_cnt += 1
        return update_cnt

    def _update_cts(self):
        self.new_cts = np.zeros_like(self.cts)      # |C| x d
        self.cts_cnt = np.zeros((self.cts.shape[0], 1)) # |C| x 1

        for i in range(self.X.shape[0]):
            id = self.cts_id[i]
            self.new_cts[id] += self.X[i, :]
            self.cts_cnt[id, 0] += 1

        self.cts = self.new_cts / self.cts_cnt

    def fit(self, X):
        self.cts = X[np.random.choice(X.shape[0], self.k), :]   # |C| x d
        self.cts_id = np.array([0 for i in range(X.shape[0])])  # N x 1
        self.X = X

        i = 0
        while True:
            update_cnt = self._update_group()
            self._update_cts()
            # print(i, self.eval())
            if update_cnt == 0:
                break
            i += 1

    def eval(self):

        err = 0.
        for i in range(self.X.shape[0]):
            x = self.X[i, :]
            c = self.cts[self.cts_id[i], :]
            err += dist(x, c)
        err /= self.X.shape[0]
        return err
