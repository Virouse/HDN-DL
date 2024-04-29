from csv import reader
import numpy as np


def load_ionosphere():
    # LOAD Data
    filename = 'datasets/ionosphere.csv'
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    dataset = np.array(dataset)
    X = dataset[:, 0:34].astype(float)
    y = dataset[:, -1].astype(int)
    return X, y
