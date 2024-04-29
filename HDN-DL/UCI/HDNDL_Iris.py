from collections import Counter
import MyLeadingTree as lt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import datetime
from HDNDL_Select import HDNDL_Select, LabelPropagation

start_t1 = datetime.datetime.now()
lt_num = 3  # 子树个数
iris = datasets.load_iris()
X = iris.data
y = iris.target
scalar = MinMaxScaler()
X = scalar.fit_transform(X)
lt1 = lt.LeadingTree(X_train=X, dc=0.5, lt_num=3)  # 整个数据集构造引领树
lt1.fit()

select_num = 6

selectInds = HDNDL_Select(lt1.gamma, y, 1, select_num)
y_select = y[selectInds]
y_predict = LabelPropagation(lt1, selectInds, y_select)
y_predict = np.delete(y_predict, selectInds)
y_test = np.delete(y, selectInds)
arr = y_predict - y_test
count = Counter(arr)[0]
print(f'Accuracy_neib:{count / len(y_test)}, {count} / {len(y_test)}')
