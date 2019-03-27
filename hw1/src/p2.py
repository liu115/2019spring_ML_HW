from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import numpy as np

X = [
        (1, 0),
        (0, 1),
        (0, -1),
        (-1, 0),
        (0, 2),
        (0, -2),
        (-2, 0),
    ]
y = [-1., -1., -1., 1., 1., 1., 1.]
N = len(X)

def dot(x1, x2):
    assert len(x1) == len(x2)
    return sum([x1[i] * x2[i] for i in range(len(x1))])

def K(x1, x2):
    return (1 + dot(x1, x2)) ** 2



## Setup quadratic parameters
Q = np.empty((N, N))
for i in range(N):
    for j in range(N):
        Q[i, j] = y[i] * y[j] * K(X[i], X[j])

Q = matrix(Q)
p = matrix(-1., (N, 1))
G = -np.eye(N)
G = G.tolist()
G = matrix(G)
h = matrix(.0, (N,1))
A = matrix(y, (1, N))
b = matrix(.0)

sol = solvers.qp(Q, p, G, h, A, b)

print(sol)
print(sol['x'])