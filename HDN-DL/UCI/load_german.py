from csv import reader
import numpy as np


def load_german():
    # LOAD Data
    filename = 'datasets/german.csv'
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    dataset = np.array(dataset)
    X = dataset[:, 0:20].astype(int)
    y = dataset[:, -1].astype(int) - 1
    return X, y
