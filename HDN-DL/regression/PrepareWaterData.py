import numpy as np
from load_data import load_data

KaixianCountyPH = load_data()

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
    # 在 Python 中，索引从 0 开始，所以这里的索引需要调整
    # 在 MATLAB 中是 Instances(i,:)= KaixianCountyPH(i:i+AttrLen,2);
    # 在 Python 中表示为 Instances[i,:] = KaixianCountyPH[i:i+AttrLen, 1]
    Instances[i, 0:6] = KaixianCountyPH[i:i + AttrLen + 1]

# # 使用 numpy 库中的 permutation 函数生成一个随机排列的序列
np.random.seed(42)
p = np.random.permutation(InstanceNum)
# # # 将样本按照 p 中的随机排列顺序划分为训练集和测试集
TrainSet = Instances[p[:10000], :]
testSet = Instances[p[10001:11001], :]
