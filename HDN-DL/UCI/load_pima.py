from csv import reader
import numpy as np


def load_pima():
    # LOAD Data
    filename = 'datasets/pima-indians-diabetes.csv'
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    dataset = np.array(dataset)
    X = dataset[:, 0:8].astype(float)
    y = dataset[:, -1].astype(int)
    return X, y
