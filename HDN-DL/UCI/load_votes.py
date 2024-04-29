from csv import reader
import numpy as np


# ？用3代替 n用0代替 y用1代替
# republican用0代替 Democracy用1代替
def load_votes():
    # LOAD Data
    filename = 'datasets/house-votes-84.csv'
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    dataset = np.array(dataset)
    X = dataset[:, 0:15].astype(int)
    y = dataset[:, -1].astype(int)
    return X, y
