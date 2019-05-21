import numpy as np

INF = 1e8

def gini_score(y):
    # Only for binary
    N = y.shape[0]
    assert N > 0
    k1 = (y == +1).astype(int).sum()
    k2 = N - k1

    gini = 1 - (k1 / N) ** 2 - (k2 / N) ** 2
    return gini

def zero_one_error(pred_y, label_y):
    assert pred_y.shape == label_y.shape
    yy = np.sign(pred_y * label_y)

    return np.sum((yy < 0).astype(np.int)) * 1. / pred_y.shape[0]

# class TreeNode(object):
#     def __init__(self, max_height=None):

#         self.max_height = max_height
    
#     def train(self):
    
#     def predict(self)

class DecisionTree(object):
    
    def __init__(self, max_height=None):
        self.max_height = max_height
        self.left = None
        self.right = None

        # Branching criteria
        self.cut = None
        self.feat_idx = None
        
        # Leaf return value
        self.is_leaf = False
        self.constant = 0


    def _fit_node(self, X, y):
        assert len(X.shape) == 2
        n_feat = X.shape[1]

        min_gini = INF
        # min_direction = None
        min_feat_idx = None
        min_cut = None
         
        for feat_idx in range(n_feat):
            part_X = X[:, feat_idx]
            part_X = np.sort(part_X)
            part_X = np.unique(part_X)
            shifted_part_X = np.concatenate([[-INF], part_X[:-1]])
            cuts = (part_X + shifted_part_X) / 2.0
            # assert cuts.shape[0] == X.shape[0]
            for cut_idx, cut in enumerate(cuts[1:]):  # Must cut one slice ?
                gini = self._gini_score(X, y, feat_idx, cut)
                # print('cut', cut)
                if gini < min_gini:
                    min_cut = cut
                    min_feat_idx = feat_idx
                    min_gini = gini

        # print('min_cut', min_cut)
        return min_feat_idx, min_cut

    def _split_tree(self, X, y, feat_idx, cut):
        left_indices = []
        right_indices = []

        for i in range(X.shape[0]):
            # print(X.shape, i, feat_idx)
            if X[i, feat_idx] < cut:
                left_indices.append(i)
            else:
                right_indices.append(i)
        
        return X[left_indices, :], X[right_indices, :], y[left_indices], y[right_indices]

    # def split_X(self, X):
    #     left_indices = []
    #     right_indices = []

    #     for i in range(X.shape[0]):
    #         # print(X.shape, i, feat_idx)
    #         if X[i, self.feat_idx] < self.cut:
    #             left_indices.append(i)
    #         else:
    #             right_indices.append(i)
    #     return X[left_indices, :], X[right_indices, :]


    def _gini_score(self, X, y, feat_idx, cut):
        _, _, left_y, right_y = self._split_tree(X, y, feat_idx, cut)
        # print(len(left_y), len(right_y), feat_idx, cut)
        left_gini = gini_score(left_y)
        right_gini = gini_score(right_y)
        # print(left_y, left_gini, right_y, right_gini)

        return left_y.shape[0] * left_gini + right_y.shape[0] * right_gini

    def fit(self, X, y):
        # if self.max_height is not None:
            # print('depth', 9999 - self.max_height) 
        # print(X, y)
        if self.max_height is not None and self.max_height == 1:
            self.is_leaf = True
            y_1 = (y == 1).astype(int).sum()
            y_2 = y.shape[0] - y_1
            self.constant = +1 if y_1 > y_2 else -1
            # print('Constant', self.constant)
            return self

        if gini_score(y) == 0:
            self.is_leaf = True
            self.constant = y[0]
            # print('Constant', self.constant)
            return self

        min_feat_idx, min_cut = self._fit_node(X, y)
        left_X, right_X, left_y, right_y = self._split_tree(
            X, y, min_feat_idx, min_cut)
        self.cut = min_cut
        self.feat_idx = min_feat_idx
        
        max_height = self.max_height - 1 if self.max_height else None
        self.left = DecisionTree(
            max_height=max_height).fit(left_X, left_y)
        self.right = DecisionTree(
            max_height=max_height).fit(right_X, right_y)
        return self
    
    def predict_one_sample(self, x):
        if self.is_leaf:
            return self.constant
        # print(self.feat_idx and self.cut)
        assert self.feat_idx is not None, "This node has not been train"
        assert self.cut is not None, "This node has not been train"

        if x[self.feat_idx] < self.cut:
            return self.left.predict_one_sample(x)
        else:
            return self.right.predict_one_sample(x)

    def predict(self, X):

        pred_y = np.zeros((X.shape[0]))
        for i in range(X.shape[0]):
            pred_y[i] = self.predict_one_sample(X[i])
        
        return pred_y

    def eval(self, X, y):
        pred_y = self.predict(X)
        return zero_one_error(pred_y, y)


def visualize_desision_tree(root):

    from graphviz import Digraph
    f = Digraph('decision_tree', filename='tree.gv')

    # f.attr('node', shape='circle')
    def run(node, i):

        if node.is_leaf:
            f.node('node' + str(i), label=str(node.constant))
            return

        f.node('node' + str(i), label='')
        cut = np.floor(node.cut * 100) / 100.0
        if node.left is not None:
            label = "x[%d] < %.2f" % (node.feat_idx, node.cut)
            f.edge('node' + str(i), 'node' + str(2*i), label=label)
            run(node.left, 2*i)
        if node.right is not None:
            label = "x[%d] ≥ %.2f" % (node.feat_idx, node.cut)
            f.edge('node' + str(i), 'node' +
                   str(2*i+1), label=label)
            run(node.right, 2*i+1)
    run(root, 1)
    f.view()


class BagTree:

    def __init__(self, bag_size=30000, sample_rate=0.8):
        self.sample_rate = sample_rate
        self.bag_size = bag_size
        self.models = []
    
    def fit(self, X, y):
        size = int(self.sample_rate * X.shape[0])
       

        for i in range(self.bag_size):
            sample_indices = np.random.randint(X.shape[0], size=size)
            sample_X = X[sample_indices, :]
            sample_y = y[sample_indices]

            print('\r{} / {}'.format(i+1, self.bag_size), end='')
            model = DecisionTree()
            model.fit(sample_X, sample_y)
            self.models.append(model)
    
    def predict(self, X):
        y_pred = np.zeros((X.shape[0]))

        for i, model in enumerate(self.models):
            y = model.predict(X)
            y_pred += y

        y_pred[y_pred >= 0] = 1
        y_pred[y_pred < 0] = -1
        return y_pred

    def eval(self, X, y):
        pred_y = self.predict(X)
        return zero_one_error(pred_y, y)
        


# f.attr(rankdir='LR', size='8,5')

# f.attr('node', shape='doublecircle')
# f.node('LR_0')
# 

# f.attr('node', shape='circle')
# f.edge('LR_0', 'LR_2', label='SS(B)')
# f.edge('LR_0', 'LR_1', label='SS(S)')

