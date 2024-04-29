import time
import numpy as np
import MyLeadingTree as lt
from load_data import load_data

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


KaixianCountyPH = load_data('KaixianCountyDO.csv')

# 原始数据集的行数
ND = len(KaixianCountyPH)
# 特征数量
AttrLen = 12
# 计算训练样本数量，每个样本含有 AttrLen 个特征和一个标签值，所以总特征数为 AttrLen+1
InstanceNum = ND - (AttrLen + 1)
# 创建一个零矩阵，用于存储处理后的样本数据
Instances = np.zeros((InstanceNum, AttrLen + 1))

# 循环遍历数据集，从第一个样本开始，逐行提取特征和标签值
for i in range(InstanceNum):
    Instances[i, 0:AttrLen + 1] = KaixianCountyPH[i:i + AttrLen + 1]

offset = 100

TrainSet = Instances[0+offset:10000+offset, :]
testSet = Instances[10001+offset:11001+offset, :]

X_train = TrainSet[:, 0:AttrLen]
y_train = TrainSet[:, AttrLen]
X_test = testSet[:, 0:AttrLen]
y_test = testSet[:, AttrLen]

lt1 = lt.LeadingTree(X_train=X_train, dc=0.001, lt_num=1)  # 整个数据集构造引领树
lt1.fit()

y_predict = np.zeros(len(y_test)) - 1

for i in range(len(y_test)):
    t1 = time.time()
    y_predict[i] = PredictLabelForNewPoint(lt1, X_test[i].reshape(1, -1), y_train)
    t2 = time.time()
    # print(f'第{i}次预测耗时{t2-t1}s')

sse = np.sum((y_test - y_predict)**2)
print(f'sse={sse},dc={lt1.dc},lt_num={lt1.lt_num}')

np.savetxt('y_pred_DO12_HDN.csv', y_predict, fmt='%.2f', delimiter=',')