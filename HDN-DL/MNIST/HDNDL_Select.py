import collections
import numpy as np


def HDNDL_Select(gamma, Y, k, l):
    # 获取类别数
    num_classes = len(np.unique(Y))
    classes = np.unique(Y)
    # 存储最终选择的样本索引
    selected_indices = np.zeros(0, dtype=int)

    # 对每个类别进行处理
    for class_label in classes:
        # 获取当前类别的样本索引
        class_indices = np.where(Y == class_label)[0]

        # 从当前类别中选择k个样本，按照gamma值降序排列
        top_k_indices = class_indices[np.argsort(gamma[class_indices])[::-1][:k]]

        # 将选出的样本索引加入最终结果
        selected_indices = np.append(selected_indices, top_k_indices)

    # 从剩余的样本中选择l - k*num_classes 个样本，按照gamma值降序排列
    if l - k * num_classes > 0:
        remaining_indices = np.setdiff1d(np.arange(len(Y)), selected_indices)
        additional_indices = remaining_indices[np.argsort(gamma[remaining_indices])[::-1][:l - k * num_classes]]

        # 将剩余选择的样本索引加入最终结果
        selected_indices = np.append(selected_indices, additional_indices)

    return selected_indices


def LabelPropagation(lt, selectInds, y):
    y_predict = np.zeros(len(lt.Pa), dtype=int) - 1
    y_predict[selectInds] = y
    queue = collections.deque()
    for i in range(len(selectInds)):
        queue.append(selectInds[i])

    while len(queue) != 0:
        Node = queue.popleft()
        indes = [i for i, x in enumerate(lt.Pa) if x == Node]

        for ind in indes:
            if ind in selectInds:
                continue
            cur_neib = lt.neib[ind]
            y_temp = y_predict[cur_neib]
            vaild_ind = np.where(y_temp != -1)[0]
            class_probs = np.bincount(y_temp[vaild_ind], weights=lt.sim[ind][vaild_ind])
            y_predict[ind] = np.argmax(class_probs)
            queue.append(ind)

    return y_predict
