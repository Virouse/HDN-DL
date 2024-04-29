import numpy as np
from sklearn.svm import SVR
from load_data import load_data

KaixianCountyPH = load_data('KaixianCountyPH.csv')

# 原始数据集的行数
ND = len(KaixianCountyPH)
# 特征数量
AttrLen = 5
# 计算训练样本数量，每个样本含有 AttrLen 个特征和一个标签值，所以总特征数为 AttrLen+1
InstanceNum = ND - (AttrLen + 1)
# 创建一个零矩阵，用于存储处理后的样本数据
Instances = np.zeros((InstanceNum, AttrLen + 1))

# 循环遍历数据集，从第一个样本开始，逐行提取特征和标签值
for i in range(InstanceNum):
    Instances[i, ] = KaixianCountyPH[i:i + AttrLen + 1]


offset = 100

TrainSet = Instances[0+offset:10000+offset, :]
testSet = Instances[10001+offset:11001+offset, :]

X_train = TrainSet[:, 0:AttrLen]
y_train = TrainSet[:, AttrLen]
X_test = testSet[:, 0:AttrLen]
y_test = testSet[:, AttrLen]

np.savetxt('y_test_PH5.csv', y_test, fmt='%.2f', delimiter=',')

lssvr = SVR(kernel='rbf', gamma=0.5, C=5, epsilon=0.1)
lssvr.fit(X_train, y_train)

y_pred = lssvr.predict(X_test)
sse = np.sum((y_test - y_pred)**2)
print(f'sse={sse}')

np.savetxt('y_pred_PH5_LSSVR.csv', y_pred, fmt='%.2f', delimiter=',')