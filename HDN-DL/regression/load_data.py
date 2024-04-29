from csv import reader
import numpy as np
# from sklearn.model_selection import train_test_split


def load_data(name):
    # LOAD Data
    # filename = 'WaterQuality/KaixianCountyDO.csv'
    filename = 'WaterQuality/' + name
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    dataset = np.array(dataset)
    X = dataset[:, 1].astype(float)
    return X
# X, y = load_yeast()
# print(y)
# print(X.shape)
# print(y.shape)
# unique_y = np.unique(y)
# for label in unique_y:
#     print(X[y == label].shape)