import numpy as np
import collections
import torch
import time

def EuclidianDist2GR(X1, X2):
    ###Using broadcasting, simpler and faster!

    # X1G= torch.from_numpy(X1)
    X1G = torch.tensor(X1, dtype=torch.float32)
    a = torch.square(X1G)##Hadamard product
    tempMG = torch.einsum('ij->i', [a]) ##列求和行，得到行
    tempMG = torch.reshape(tempMG, (-1, 1))
    # print(f'tempMG数据类型为{tempMG.dtype}')

    # X2G = torch.from_numpy(X2)
    X2G = torch.tensor(X2, dtype=torch.float32)
    b = torch.square(X2G)  ##Hadamard product
    tempNG = torch.einsum('ij->i', [b])  ##行求和，得到列向量
    # print(f'tempNG数据类型为{tempNG.dtype}')

    ##MNSum = tempMG + tempNG
    X2GT = torch.einsum('ij->ji', [X2G])
    sqdistG = tempMG + tempNG - 2 * torch.einsum('ik,kj->ij', [X1G, X2GT])

    # print(f'sqdistG数据类型为{sqdistG.dtype}')


    sqdistG[sqdistG < 0] = 0
    resultG = torch.sqrt(sqdistG)
    resultG = resultG.cpu().numpy()

    # print(f'resultG数据类型为{resultG.dtype}')

    return resultG

def ComputeLocalDensityR(D, dc):
    t1 = time.time()
    DG = torch.tensor(D, dtype=torch.float32)
    tempMat1 = torch.exp(-(torch.square(DG)))
    tempMat2 = torch.pow(tempMat1, dc ** (-2))
    density1 = torch.einsum('ij->i', tempMat2)
    density = density1.cpu().numpy() - 1
    t2 = time.time()
    # print(f'density耗时{t2-t1}')
    # print(f'density数据类型为{density.dtype}')

    return density




class LeadingTree:
    """
    Leading Tree
    """
    def __init__(self, X_train, dc, lt_num, D):
        N = len(X_train)
        self.X_train = X_train
        self.dc = dc
        self.lt_num = lt_num
        self.D = D
        self.density = np.zeros(N, dtype='float32')
        self.Pa = np.zeros(N, dtype=int)
        self.delta = np.zeros(N, dtype='float32')
        self.gamma = np.zeros(N, dtype='float32')
        self.gamma_D = np.zeros(N, dtype=int)
        self.Q = np.zeros(N, dtype=int)
        self.AL = [np.zeros((0, 1), dtype=int) for i in range(lt_num)]  # AL[i] store all indexes of a subtree
        self.layer = np.zeros(len(X_train), dtype=int)
        self.neib = [np.zeros((0, 1), dtype=int) for i in range(len(X_train))]
        self.sim = [np.zeros((0, 1), dtype=int) for i in range(len(X_train))]

    def ComputeParentNode(self, D, Q):
        """
        Calculate the distance to the nearest data point of higher density (delta) and the parent node (Pa)
        :param D: The Euclidean distance of all samples
        :param Q:Sort by index in descending order of sample local density
        :return:
        self.delta: the distance of the sample to the closest data point with a higher density
        self.Pa: the index of the parent node of the sample
        """

        for i in range(len(Q)):
            if i == 0:
                self.delta[Q[i]] = max(D[Q[i]])
                self.Pa[Q[i]] = -1
            else:
                greaterInds = Q[0:i]
                D_A = D[Q[i], greaterInds]
                self.delta[Q[i]] = min(D_A)
                self.Pa[Q[i]] = greaterInds[np.argmin(D_A)]

    def ProCenter(self, density, delta, Pa):
        """
        Calculate the probability of being chosen as the center node and Disconnect the Leading Tree
        :param density: local density of all samples
        :param delta: the distance of the sample to the closest data point with a higher density
        :param Pa: the index of the parent node of the sample
        :return:
        self.gamma: the probability of the sample being chosen as a center node
        self.gamma_D: Sort the gamma index in descending order
        """
        self.gamma = density * delta
        self.gamma_D = np.argsort(self.gamma)[::-1]

        # Disconnect the Leading Tree
        for i in range(self.lt_num):
            Pa[self.gamma_D[i]] = -1

    def GetSubtreeR(self, gamma_D, lt_num, Q, pa):
        """
         Subtree
        :param gamma_D:
        :param lt_num: the number of subtrees
        :return:
        self.AL: AL[i] store indexes of a subtrees, i = {0, 1, ..., lt_num-1}
        """
        for i in range(lt_num):
            self.AL[i] = np.append(self.AL[i], gamma_D[i])

        N = len(gamma_D)
        treeID = np.zeros((N,1),dtype=int)-1
        for i in range(lt_num):
            treeID[gamma_D[i]]=i

        for nodei in range(N): ### casscade label assignment
            curInd = Q[nodei]
            if treeID[curInd]>-1:
                continue

            else:
                paID = pa[curInd]
                self.layer[curInd] = self.layer[paID]+1
                curTreeID = treeID[paID]
                treeID[curInd] = curTreeID
                self.AL[curTreeID[0]] = np.append(self.AL[curTreeID[0]], curInd)

    def ComputeNeiborBydensity(self, D, Q):
        for i in range(len(Q)):
            if self.Pa[Q[i]] == -1:
                continue
            else:
                greaterInds = Q[0:i]
                D_A = D[Q[i], greaterInds]
                # n = self.layer[Q[i]] - 1
                if self.layer[Q[i]] <= 4:
                    n = self.layer[Q[i]] - 1
                else:
                    n = 4
                for j in range(n):
                    cur_min = np.argmin(D_A)
                    self.neib[Q[i]] = np.append(self.neib[Q[i]], greaterInds[cur_min])
                    greaterInds = np.delete(greaterInds, cur_min)
                    D_A = np.delete(D_A, cur_min)
                # index = np.argsort(D_A)[0:self.layer[Q[i]] - 1]
                # neib = greaterInds[index]
                # self.neib[Q[i]] = np.append(self.neib[Q[i]], neib)

    def ComputeDeltaByNeibor(self, neib, D):
        self.delta_neib = np.zeros(len(neib))
        for i in range(len(neib)):
            cur_neib = neib[i]
            if len(cur_neib) == 0:
                continue
            distance_neib = D[i, cur_neib]
            distance_avg = np.sum(distance_neib) / len(distance_neib)
            self.delta_neib[i] = distance_avg
            sim = distance_avg / distance_neib
            self.sim[i] = np.append(self.sim[i], sim)


    def Edges(self, Pa):  # store edges of subtrees
        """

        :param Pa:  the index of the parent node of the sample
        :return:
        self. edges: pairs of child node and parent node
        """
        edgesO = np.array(list(zip(range(len(Pa)), Pa)))
        ind = edgesO[:, 1] > -1
        self.edges = edgesO[ind,]

    def fit(self):
        t1 = time.time()
        self.density = ComputeLocalDensityR(self.D, self.dc)
        self.Q = np.argsort(self.density)[::-1]
        t2 = time.time()
        # print(f'密度计算完毕,耗时{t2-t1}')
        t1 = time.time()
        self.ComputeParentNode(self.D, self.Q)
        t2 = time.time()
        # print(f'父节点计算完毕,耗时{t2-t1}')
        self.ProCenter(self.density, self.delta, self.Pa)

        # print('子树划分....')

        t1 = time.time()
        self.GetSubtreeR(self.gamma_D, self.lt_num, self.Q, self.Pa)
        t2 = time.time()
        # print('GetSubtreeR耗时', (t2 - t1))
        # self.Edges(self.Pa)
        self.layer = self.layer + 1
        t3 = time.time()
        self.ComputeNeiborBydensity(self.D, self.Q)
        t4 = time.time()
        self.ComputeDeltaByNeibor(self.neib, self.D)
        t5 = time.time()

