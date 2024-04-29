from csv import reader
import numpy as np

def LoadAndProc():
    # pathName = 'D:/GZU2022b/LMLF/venv/data/'
    pathName = 'datasets/'
    filenames = [pathName + 'letter.csv']
    dim = 17
    DataSet = np.zeros((0, dim))
    # Letters= [chr(i) for i in range(65,91)]
    for filename in filenames:
        with open(filename, 'r') as file:
            dataset = list()
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        dataset = np.array(dataset)
        DataSet = np.append(DataSet, dataset, axis=0)
    X = DataSet[:, 1:dim].astype(float)
    Y = DataSet[:, 0].astype(int)


    ###mapping the labels
    # Unilabels = np.unique(Y)
    # for i in range(len(Y)):
    #     Y[i] = int(list(Unilabels).index(Y[i]))
    # Y = np.array(Y).astype(int)
    return X, Y

def LoadAndProcLT(ltSN):
    pathName = 'datasets/'
    filenames = [pathName + 'LT'+str(ltSN)+'.csv']
    dim = 17
    DataSet = np.zeros((0, dim))
    # Letters= [chr(i) for i in range(65,91)]
    for filename in filenames:
        with open(filename, 'r') as file:
            dataset = list()
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        dataset = np.array(dataset)
        DataSet = np.append(DataSet, dataset, axis=0)
    X = DataSet[:, 1:dim].astype(float)
    C = DataSet[:, 0].astype(float)
    Y = np.zeros(len(C), dtype=int)

    for i in range(len(Y)):
        Y[i] = C[i]

    ###mapping the labels
    # Unilabels = np.unique(Y)
    # for i in range(len(Y)):
    #     Y[i] = int(list(Unilabels).index(Y[i]))
    # Y = np.array(Y).astype(int)
    return X, Y


# def LoadAndMerge():
#     pathName = 'D:/GZU2022/research/DeLaLA-Manuscript/Figures/IllustratingFigure/'
#     filenames = [pathName + '2and3.csv', pathName + '3and8.csv', pathName + '8and3.csv']
#     dim = 785
#     DataSet = np.zeros((0, dim))
#     # Letters= [chr(i) for i in range(65,91)]
#     for filename in filenames:
#         with open(filename, 'r') as file:
#             dataset = list()
#             csv_reader = reader(file)
#             for row in csv_reader:
#                 if not row:
#                     continue
#                 dataset.append(row)
#         dataset = np.array(dataset)
#         DataSet = np.append(DataSet, dataset, axis=0)
#     X = DataSet[:, :dim - 1].astype(float)
#     Y = DataSet[:, dim - 1]
#     Y = [int(float(Y[i])) for i in range(len(Y))]

    ###mapping the labels
    # Unilabels = np.unique(Y)
    # for i in range(len(Y)):
    #     Y[i] = int(list(Unilabels).index(Y[i]))

    # return X, Y

# def LoadAndProc(classList):
#     filename = 'D:/PycharmProjects/LMLF/venv/data/letter.csv'
#     dataset = list()
#     # Letters= [chr(i) for i in range(65,91)]
#     with open(filename, 'r') as file:
#         csv_reader = reader(file)
#         for row in csv_reader:
#             if not row:
#                 continue
#             dataset.append(row)
#     dataset = np.array(dataset)
#     X = dataset[:, 1:].astype(float)
#     yletter = dataset[:, 0]
#     Y =np.array([ord(c)-65 for c in yletter])
#     YsubInds = [i for i in range(len(Y)) if Y[i] in classList]
#     X= X[YsubInds]
#     Y = Y[YsubInds]
#     for i in range (len(Y)):
#         temp = Y[i]
#         Y[i]=list(classList).index(temp)
#     return X,Y
