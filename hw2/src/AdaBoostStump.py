import numpy as np

INF = 1e8

def zero_one_error(pred_y, label_y):
    assert pred_y.shape == label_y.shape
    yy = np.sign(pred_y * label_y)

    return np.sum((yy < 0).astype(np.int)) * 1. / pred_y.shape[0]

def weighted_zero_one_error(pred_y, label_y, weight):
    assert pred_y.shape == label_y.shape
    assert pred_y.shape == weight.shape
    yy = np.sign(pred_y * label_y)
    return np.sum((yy < 0).astype(np.int) * weight) / np.sum(weight)


class g(object):

    def __init__(self):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError


class DecisionStump(g):

    def __init__(self):
        self.feature_idx = None
        self.s = None
        self.cut = None

    def fit(self, X, y, X_weight=None):
        n_feature = X.shape[1]
        N = X.shape[0]
        min_err = INF
        min_s = None
        min_cut = None
        min_feature_idx = None

        for i in range(n_feature):
            part_X = X[:, i]
            part_X = np.sort(part_X)
            shifted_part_X = np.concatenate([[-INF], part_X[:-1]])
            cuts = (part_X + shifted_part_X) / 2.0
            assert cuts.shape[0] == X.shape[0]

            for j, cut in enumerate(cuts):

                for s in (+1, -1):
                    err = self._score(X, y, X_weight, s, i, cut)
                    if err < min_err:
                        min_s = s
                        min_feature_idx = i
                        min_cut = cut
                        min_err = err
        self.s = min_s
        self.feature_idx = min_feature_idx
        self.cut = min_cut

    def _score(self, X, y, X_weight, s, feature_idx, cut):
        pred_y = self._predict(X, s, feature_idx, cut)
        if X_weight is not None:
            error = weighted_zero_one_error(pred_y, y, X_weight)
        else:
            error = zero_one_error(pred_y, y)

        return error

    def _predict(self, X, s, feature_idx, cut):
        if s > 0:
            # if s = +1
            pred_y = 2 * (X[:, feature_idx] >= cut).astype(np.int) - 1
        else:
            # if s = -1
            pred_y = 2 * (X[:, feature_idx] < cut).astype(np.int) - 1
        return pred_y


    def predict(self, X):
        '''
        X[i, feature_idx] > cuts = +s
        X[i, feature_idx] < cuts = -s
        return pred_y
        '''

        if self.feature_idx is None \
                or self.s is None \
                or self.cut is None:
            raise Exception("Model should be fitted data before predict")

        return self._predict(X, self.s, self.feature_idx, self.cut)

    def score(self, X, y, X_weight=None):
        if self.feature_idx is None \
                or self.s is None \
                or self.cut is None:
            raise Exception("Model should be fitted data before score")
        return self._score(X, y, X_weight, self.s, self.feature_idx, self.cut)

    def __repr__(self):
        return 'Decision Stump with s={} feature_idx={} cut={}'\
                .format(self.s, self.feature_idx, self.cut)


class AdaBoost(object):

    def __init__(self, max_iter=300, g_class=DecisionStump):
        self.max_iter = max_iter
        self.g_list = []
        self.g_weights = []
        self.g_class = g_class
        self.weight_list = []

    def _update_weight(self, X, y, X_weight, g):
        '''
        g is the function predicting X

        return (new X_weight) and (scaling factor)
        '''

        error = g.score(X, y, X_weight)
        scale_factor = np.sqrt((1 - error) / error)
        new_weight = np.copy(X_weight)

        pred_y = g.predict(X)

        for i in range(y.shape[0]):
            if pred_y[i] * y[i] >= 0:
                # correct
                new_weight[i] /= scale_factor
            else:
                # incorrect
                new_weight[i] *= scale_factor

        return new_weight, scale_factor

    def fit(self, X, y):
        N = X.shape[0]
        n_features = X.shape[1]
        X_weight = np.ones((N)) / N

        for i in range(self.max_iter):
            self.weight_list.append(X_weight)
            current_g = self.g_class()
            current_g.fit(X, y, X_weight)
            # print('iter={}'.format(i+1), current_g)
            self.g_list.append(current_g)
            X_weight, scale_factor = self._update_weight(X, y, X_weight, current_g)

            self.g_weights.append(np.log(scale_factor))

    def predict(self, X, first_n_g=None):
        '''
        G(x) = sign( \sum alpha_t g_t(x) )
        '''
        pred_y = np.zeros((X.shape[0]))
        g_list = self.g_list if not first_n_g else self.g_list[:first_n_g]
        for i, g in enumerate(g_list):
            g_pred_y = g.predict(X)
            pred_y += g_pred_y * self.g_weights[i]

        pred_y = np.sign(pred_y)
        return pred_y

    def score(self, X, y, first_n_g=None):

        pred_y = self.predict(X, first_n_g)
        return zero_one_error(pred_y, y)
