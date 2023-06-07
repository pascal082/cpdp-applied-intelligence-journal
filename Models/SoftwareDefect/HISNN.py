import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier


class HISNN(object):

    def __init__(self, MinHam=1, n_neighbors=5,
                 KNNneighbors=5):
        self.MinHam = MinHam
        self.neighbors = n_neighbors
        self.m = KNeighborsClassifier(n_neighbors=KNNneighbors)


    def _MahalanobisDist(self, data, base):

        covariance = np.cov(base.T)  # calculate the covarince matrix
        inv_covariance = np.linalg.pinv(covariance)
        mean = np.mean(base, axis=0)
        dist = np.zeros((np.asarray(data)).shape[0])
        for i in range(dist.shape[0]):
            dist[i] = distance.mahalanobis(data[i], mean, inv_covariance)
        return dist


    def TrainInstanceFiltering(self, Xsource, Ysource, Xtarget, Ytarget):
        # source outlier remove based on source
        dist = self._MahalanobisDist(Xsource, Xsource)
        threshold = np.mean(dist) * 3 * np.std(dist)
        outliers = []
        for i in range(len(dist)):
            if dist[i] > threshold:
                outliers.append(i)  # index of the outlier
        Xsource = np.delete(Xsource, outliers, axis=0)
        Ysource = np.delete(Ysource, outliers, axis=0)

        # source outlier remove based on target
        dist = self._MahalanobisDist(Xsource, Xtarget)
        threshold = np.mean(dist) * 3 * np.std(dist)
        outliers = []
        for i in range(len(dist)):
            if dist[i] > threshold:
                outliers.append(i)  # index of the outlier
        Xsource = np.delete(Xsource, outliers, axis=0)
        Ysource = np.delete(Ysource, outliers, axis=0)

        # NN filter for source data based on target
        neigh = NearestNeighbors(radius=self.MinHam, metric='hamming')
        neigh.fit(Xsource)

        filtered = []
        for item in Xtarget:
            rng = neigh.radius_neighbors(item.reshape(1, -1))
            indexNeighs = rng[1][0]
            for it in indexNeighs:
                if it not in filtered:
                    filtered.append(it)

        a = np.zeros((len(filtered), Xsource.shape[1]))
        b = np.zeros(len(filtered))
        for i in range(len(filtered)):
            a[i] = Xsource[filtered[i]]
            b[i] = Ysource[filtered[i]]

        Xsource = a
        Ysource = b
        return Xsource, Ysource,Xtarget, Ytarget

    


