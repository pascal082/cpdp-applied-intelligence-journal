from sklearn.neighbors import NearestNeighbors
import numpy as np


class Bruakfilter(object):
    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors

    def run(self, Xsource, Ysource, Xtarget, Ytarget):
        Xsource = np.log(Xsource + 1)
        Xtarget = np.log(Xtarget + 1)

        if self.n_neighbors > Xsource.shape[0]:
            return 0, 0, 0, 0

        knn = NearestNeighbors()
        knn.fit(Xsource)
        data = []
        ysel = []

        for item in Xtarget:
            tmp = knn.kneighbors(item.reshape(1, -1), self.n_neighbors, return_distance=False)
            tmp = tmp[0]
            for i in tmp:
                if list(Xsource[i]) not in data:
                    data.append(list(Xsource[i]))
                    ysel.append(Ysource[i])
        Xsource = np.asanyarray(data)
        Ysource = np.asanyarray(ysel)

        return Xsource, Ysource, Xtarget, Ytarget



