from csv import reader
import numpy as np


def load_Monkl():
    # LOAD Data
    filename = 'datasets/Monkl.csv'
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    dataset = np.array(dataset)
    X = dataset[:, 1:7].astype(int)
    y = dataset[:, 0].astype(int)
    return X, y
