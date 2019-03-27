import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm
X = [
    (1, 0),
    (0, 1),
    (0, -1),
    (-1, 0),
    (0, 2),
    (0, -2),
    (-2, 0),
]

y = [-1, -1, -1, 1, 1, 1, 1]

#### Problem 1
def t(x):
    x1, x2 = x
    z1 = 2 * (x2**2) - 4 * x1 + 2
    z2 = x1 ** 2 - 2 * x2 - 3
    return (z1, z2)

tx = [t(_x) for _x in X]
pxx = []
pyy = []
nxx = []
nyy = []
for i in range(7):
    print(X[i][0], X[i][1], tx[i][0], tx[i][1], y[i])
    if y[i] > 0:
        pxx += [tx[i][0]]
        pyy += [tx[i][1]]
    else:
        nxx += [tx[i][0]]
        nyy += [tx[i][1]]
plt.scatter(pyy, pxx, c='blue')
plt.scatter(nyy, nxx, c='red')
plt.axis([-10, 5, -5, 11])
plt.xlabel('z_2')
plt.ylabel('z_1')

x = np.arange(-10, 10)
_y = 5 * np.ones_like(x)
plt.plot(x, _y, c='black', linestyle='--')
plt.savefig('p1.png')
# plt.show()

#### Problem 2
clf = svm.SVC(kernel='poly', C=10000, degree=2, gamma=1, coef0=1)
X = [list(_x) for _x in X]

print(X)
print(y)
clf.fit(X, y)
print(clf.support_)
print(clf.dual_coef_)
