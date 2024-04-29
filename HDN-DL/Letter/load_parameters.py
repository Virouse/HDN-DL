from csv import reader
import numpy as np
# from sklearn.model_selection import train_test_split


def load_parameters(paramaters):
    # LOAD Data
    filename = paramaters
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    dataset = np.array(dataset)
    dc = dataset[:, 0].astype(float)
    lt_num = dataset[:, 1].astype(int)
    k = dataset[:, 2].astype(int)
    l = dataset[:, 3].astype(int)
    return dc, lt_num, k, l
# X, y = load_yeast()
# print(y)
# print(X.shape)
# print(y.shape)
# unique_y = np.unique(y)
# for label in unique_y:
#     print(X[y == label].shape)