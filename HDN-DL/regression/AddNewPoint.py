import numpy as np
from sklearn import datasets
import MyLeadingTree as lt

def EuclidianDistsq(X1, X2):
    ###Using broadcasting, simpler and faster!
    tempM = np.sum(X1 ** 2, 1).reshape(-1, 1)  ##行数不知道，只知道列数为1
    tempN = np.sum(X2 ** 2, 1)  # X2 ** 2: element-wise square, sum(_,1): 沿行方向相加，但最后是得到行向量
    sqdist = tempM + tempN - 2 * np.dot(X1, X2.T)
    sqdist[sqdist < 0] = 0
    return sqdist


def PredictLabelForNewPoint(LT, x, y):
    ind = len(LT.X_train)
    dist2_x = EuclidianDistsq(LT.X_train, x)
    density = LT.density
    for i in range(len(LT.X_train)):
        density[i] += np.exp(-(dist2_x[i]/LT.dc**2))
    density_x = np.sum(np.exp(-(dist2_x/LT.dc**2)))
    density = np.append(LT.density, density_x)
    Q = np.argsort(density)[::-1]

    # computer parentNode and neibor
    if Q[0] != ind:
        indInQ = np.where(Q == ind)[0][0]
        greaterInds = Q[0:indInQ]
        distHighDensity = dist2_x[greaterInds]
        Pa = greaterInds[np.argmin(distHighDensity)]
        return y[Pa]
    else:
        chind = np.argmin(dist2_x)
        return y[chind]
        # layer = LT.layer[Pa] + 1
        # if layer <= 4:
        #     n = layer - 1
        # else:
        #     n = 4
        # neib = np.zeros(0, dtype=int)
        # delta = np.zeros(0, dtype=float)
        # for j in range(n):
        #     cur_min = np.argmin(distHighDensity)
        #     neib = np.append(neib, greaterInds[cur_min])
        #     delta = np.append(delta, distHighDensity[cur_min])
        #     greaterInds = np.delete(greaterInds, cur_min)
        #     distHighDensity = np.delete(distHighDensity, cur_min)
        #
        # for j in range(len(delta)):
        #     if delta[j] == 0:
        #         delta[j] = 1e-8
        #
        # # predict label by neib
        # dist_avg = np.sum(delta) / len(delta)
        # sim = dist_avg / delta
        #
        # y_train = y[neib]
        # class_probs = np.bincount(y_train, weights=sim)
        # y_predict = np.argmax(class_probs)
        # return y_predict


iris = datasets.load_iris()
X = iris.data
y = iris.target
lt1 = lt.LeadingTree(X_train=X, dc=0.2, lt_num=3)  # 整个数据集构造引领树
lt1.fit()

x_new = np.array([4.9, 3.0, 1.4, 0.2]).reshape(1, 4)
y_predict = PredictLabelForNewPoint(lt1, x_new, y)

X_add = np.append(X, x_new, axis=0)
lt2 = lt.LeadingTree(X_train=X_add, dc=0.2, lt_num=3)
lt2.fit()