from csv import reader
import numpy as np


def load_yeast():
    # LOAD Data
    filename = 'datasets/Yeast.csv'
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    dataset = np.array(dataset)
    X = dataset[:, 1:9].astype(float)
    y = dataset[:, -2].astype(int) - 1
    return X, y
