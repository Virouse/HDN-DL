import collections
import numpy as np


def HDNDL_Select(gamma, Y, k, l):
    num_classes = len(np.unique(Y))

    selected_indices = []

    for class_label in range(num_classes):
        class_indices = np.where(Y == class_label)[0]

        top_k_indices = class_indices[np.argsort(gamma[class_indices])[::-1][:k]]

        selected_indices.extend(top_k_indices.tolist())

    if l - k * num_classes > 0:
        remaining_indices = np.setdiff1d(np.arange(len(Y)), selected_indices)
        additional_indices = remaining_indices[np.argsort(gamma[remaining_indices])[::-1][:l - k * num_classes]]

        selected_indices.extend(additional_indices.tolist())

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
