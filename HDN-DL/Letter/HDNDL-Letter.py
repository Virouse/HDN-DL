from collections import Counter
import MyLeadingTree as lt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
from PreprocessLetter import LoadAndProc
from load_parameters import load_parameters
from HDNDL_Select import HDNDL_Select, LabelPropagation


def EuclidianDist2(X1, X2):
    ###Using broadcasting, simpler and faster!
    tempM = np.sum(X1 ** 2, 1, dtype='float32').reshape(-1, 1)  ##行数不知道，只知道列数为1
    tempN = np.sum(X2 ** 2, 1, dtype='float32')  # X2 ** 2: element-wise square, sum(_,1): 沿行方向相加，但最后是得到行向量
    sqdist = tempM + tempN - 2 * np.dot(X1, X2.T).astype('float32')
    sqdist[sqdist < 0] = 0

    return np.sqrt(sqdist)


lt_num = 45  # 子树个数
X, y = LoadAndProc()
scalar = MinMaxScaler()
X = scalar.fit_transform(X)
t1 = time.time()
D = EuclidianDist2(X, X)
lt1 = lt.LeadingTree(X_train=X, dc=0.14, lt_num=lt_num, D=D)  # 整个数据集构造引领树  0.14
lt1.fit()

y_predict_all = np.zeros(len(y), dtype=int) - 1
selectInds_all = np.zeros(0, dtype=int)
removeInds = np.zeros(0, dtype=int)

# #  参数设置
#
#  第一层
order1 = 0
dc_1, lt_num_1, k_1, l_1 = load_parameters('parameters_order1.csv')

#  第二层
order2 = 0
dc_2, lt_num_2, k_2, l_2 = load_parameters('parameters_order2.csv')

#  lt2
order_lt2 = 0
dc_lt2, lt_num_lt2, empty1, empty2 = load_parameters('parameters_lt2.csv')

#  lt3

order_lt3 = 0
dc_lt3, lt_num_lt3, empty3, empty4 = load_parameters('parameters_lt3.csv')

for i in range(lt_num):
    train_AL = lt1.AL[i]
    y_AL = y[lt1.AL[i]]
    label_num = len(np.unique(y[train_AL]))

    if label_num == 1:
        selectInds_all = np.append(selectInds_all, train_AL[0])
        y_predict_all[train_AL] = y[train_AL[0]]
        arr = y_predict_all[train_AL] - y[train_AL]
        count1 = Counter(arr)[0]
        # print(f'第一层case1:子树{i}的类别数为{len(np.unique(y[train_AL]))},'
        #       f'准确率为{(count1 - 1) / (len(arr) - 1)}, {count1 - 1}/{(len(arr) - 1)}')
    elif 2 <= label_num <= 3:
        D_A = D[train_AL]
        D_A = D_A[:, train_AL]
        LT = lt.LeadingTree(X_train=X[train_AL], dc=dc_1[order1], lt_num=lt_num_1[order1], D=D_A)
        LT.fit()
        selectInds_local = HDNDL_Select(LT.gamma, y_AL, k_1[order1], l_1[order1])
        selectInds_universe = train_AL[selectInds_local]
        selectInds_all = np.append(selectInds_all, selectInds_universe)
        y_predict_all[train_AL] = LabelPropagation(LT, selectInds_local, y_AL[selectInds_local])
        arr = y_predict_all[train_AL] - y[train_AL]
        count1 = Counter(arr)[0]
        # print(f'第一层case2:子树{i}类别数为{len(np.unique(y_AL))},'
        #       f'准确率为{(count1 - len(selectInds_local)) / (len(arr) - len(selectInds_local))}, {count1 - len(selectInds_local)}/{(len(arr) - len(selectInds_local))}'
        #       f'*****dc={LT.dc}, lt_num={LT.lt_num}, k={k_1[order1]}, l={l_1[order1]}')
        order1 += 1
    else:
        D_2 = D[train_AL]
        D_2 = D_2[:, train_AL]
        lt2 = lt.LeadingTree(X_train=X[train_AL], dc=dc_lt2[order_lt2], lt_num=lt_num_lt2[order_lt2], D=D_2)
        lt2.fit()
        # print(f'子树{i}, lt2_dc={dc_lt2[order_lt2]}, lt2_lt_num={lt_num_lt2[order_lt2]}')
        order_lt2 += 1

        for j in range(lt2.lt_num):
            train_AL_2 = train_AL[lt2.AL[j]]
            y_AL_2 = y[train_AL[lt2.AL[j]]]
            label_num = len(np.unique(y[train_AL_2]))
            if label_num == 1:
                y_predict_all[train_AL_2] = y[train_AL_2[0]]
                selectInds_all = np.append(selectInds_all, train_AL_2[0])
                arr = y_predict_all[train_AL_2] - y[train_AL_2]
                count1 = Counter(arr)[0]
                # print(f'第二层case1:子树{i}-{j}的类别数为{len(np.unique(y[train_AL_2]))},'
                #       f'准确率为{(count1 - 1) / (len(arr) - 1)}, {count1 - 1}/{len(arr) - 1}')

            elif 2 <= label_num <= 3:
                D_A = D[train_AL_2]
                D_A = D_A[:, train_AL_2]
                LT = lt.LeadingTree(X_train=X[train_AL_2], dc=dc_2[order2], lt_num=lt_num_2[order2], D=D_A)
                LT.fit()
                selectInds_local = HDNDL_Select(LT.gamma, y_AL_2, k_2[order2], l_2[order2])
                selectInds_universe = train_AL_2[selectInds_local]
                selectInds_all = np.append(selectInds_all, selectInds_universe)
                y_predict_all[train_AL_2] = LabelPropagation(LT, selectInds_local, y_AL_2[selectInds_local])
                arr = y_predict_all[train_AL_2] - y[train_AL_2]
                count1 = Counter(arr)[0]
                # print(f'第二层case2:子树{i}-{j}类别数为{len(np.unique(y_AL_2))}, '
                #       f'准确率为{(count1 - len(selectInds_local)) / (len(arr) - len(selectInds_local))}, {count1 - len(selectInds_local)}/{(len(arr) - len(selectInds_local))}'
                #       f'*****dc={LT.dc}, lt_num={LT.lt_num}, k={k_2[order2]}, l={l_2[order2]}, order2 ={order2}')
                order2 += 1
            else:
                D_3 = D[train_AL_2]
                D_3 = D_3[:, train_AL_2]
                lt3 = lt.LeadingTree(X_train=X[train_AL_2], dc=dc_lt3[order_lt3], lt_num=lt_num_lt3[order_lt3], D=D_3)
                lt3.fit()
                order_lt3 += 1
                for k in range(lt3.lt_num):
                    train_AL_3 = train_AL_2[lt3.AL[k]]
                    y_AL_3 = y[train_AL_2[lt3.AL[k]]]
                    label_num = len(np.unique(y[train_AL_3]))
                    if len(train_AL_3) < 5:
                        removeInds = np.append(removeInds, train_AL_3)
                    elif label_num == 1:
                        y_predict_all[train_AL_3] = y[train_AL_3[0]]
                        selectInds_all = np.append(selectInds_all, train_AL_3[0])
                        arr = y_predict_all[train_AL_3] - y[train_AL_3]
                        count1 = Counter(arr)[0]
                        # print(f'第三层case1:子树{i}-{j}-{k}的类别数为{len(np.unique(y[train_AL_3]))},'
                        #       f'准确率为{(count1 - 1) / (len(arr) - 1)}, {count1 - 1}/{len(arr) - 1}')
                    else:
                        D_A = D[train_AL_3]
                        D_A = D_A[:, train_AL_3]
                        LT = lt.LeadingTree(X_train=X[train_AL_3], dc=0.15, lt_num=3, D=D_A)
                        LT.fit()
                        selectInds_local = HDNDL_Select(LT.gamma, y_AL_3, 1, label_num + 1)
                        selectInds_universe = train_AL_3[selectInds_local]
                        selectInds_all = np.append(selectInds_all, selectInds_universe)
                        y_predict_all[train_AL_3] = LabelPropagation(LT, selectInds_local, y_AL_3[selectInds_local])
                        arr = y_predict_all[train_AL_3] - y[train_AL_3]
                        count1 = Counter(arr)[0]
                        # print(f'第三层case2:子树{i}-{j}-{k}类别数为{len(np.unique(y[train_AL_3]))}, '
                        #       f'准确率为{(count1 - len(selectInds_local)) / (len(arr) - len(selectInds_local))}, {count1 - len(selectInds_local)}/{(len(arr) - len(selectInds_local))}'
                        #       f'*****dc={LT.dc}, lt_num={LT.lt_num}, k={2}, l={label_num}, order2 ={order2}')

y_predict_all = np.delete(y_predict_all, selectInds_all)
y_test_all = np.delete(y, selectInds_all)

arr = y_predict_all - y_test_all
count1 = Counter(arr)[0]
t3 = time.time()
print(f'select_num:{len(selectInds_all)},Accuracy:{count1 / len(y_test_all)}, {count1}/{len(y_test_all)}')
print(f'总耗时{t3 - t1}')
