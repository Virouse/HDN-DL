from collections import Counter
import MyLeadingTree as lt
import numpy as np
from HDNDL_Select import HDNDL_Select, LabelPropagation
from sklearn.preprocessing import MinMaxScaler
import datetime
from load_yeast import load_yeast


start_t1 = datetime.datetime.now()
lt_num = 87  # 子树个数
X, y = load_yeast()
scalar = MinMaxScaler()
X = scalar.fit_transform(X)
lt1 = lt.LeadingTree(X_train=X, dc=0.0605, lt_num=lt_num)  # 整个数据集构造引领树
lt1.fit()

l = 111

selectInds = HDNDL_Select(lt1.gamma, y, 4, l)
y_select = y[selectInds]
layer_select = lt1.layer[selectInds]
y_predict = np.zeros(len(y), dtype=int) - 1
y_predict[selectInds] = y_select
y_predict = LabelPropagation(lt1, selectInds, y_select)
y_predict = np.delete(y_predict, selectInds)
y_test = np.delete(y, selectInds)
arr = y_predict - y_test
count = Counter(arr)[0]
print(f'Accuracy_neib:{count / len(y_test)}, {count} / {len(y_test)}')

