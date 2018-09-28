import numpy as np
import matplotlib.pyplot as plt
from Laplacian import *
from SimilarityFusion import *
import numpy.linalg as linalg
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def getSSM(X):
    """
    Compute a Euclidean self-similarity image between a set of points
    :param X: An Nxd matrix holding the d coordinates of N points
    :return: An NxN self-similarity matrix
    """
    D = np.sum(X**2, 1)[:, None]
    D = D + D.T - 2*X.dot(X.T)
    D[D < 0] = 0
    D = 0.5*(D + D.T)
    D = np.sqrt(D)
    return D

def getSlidingWindow(x, dim):
    N = x.size
    M = N - dim + 1
    X = np.zeros((M, dim))
    for k in range(dim):
        X[:, k] = x[k:k+M]
    return X


if __name__ == '__main__':
    locs_left_of_gap = 5737-1
    locs_right_of_gap = 5864
    x = np.loadtxt('MIR_Arrhythmea/ecgRR.txt')
    x = np.concatenate((x[0:locs_left_of_gap], x[locs_right_of_gap::]))
    x = x[0:1000]
    print("np.min(x) = ", np.min(x))
    # NOTE: Early works in EEG used winLength of 300
    winLength = 50
    X = getSlidingWindow(x[1::]/x[0:-1], winLength)
    D = getSSM(X)
    #W = getW(D, 50, Mu = 0.5)
    W = np.max(D) - D
    np.fill_diagonal(W, 0)
    print("np.min(W) = ", np.min(W))
    w, v = linalg.eigh(D)
    v = v[:, 0:10]
    thisv = v[:, 0]

    n_clusters = 3
    pca = PCA(n_components=10)
    Y = pca.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(Y)
    idx = kmeans.labels_[0:thisv.size]


    x = x[0:thisv.size]

    plt.subplot2grid((1, 4), (0, 0), colspan=2)
    for k in range(n_clusters):
        xk = np.array(x)
        xk[np.abs(idx-k) > 0] = np.nan
        plt.plot(xk)
    plt.title("Window Length = %i, Total time in seconds: %.3g"%(winLength, np.sum(x)/120.0))

    plt.subplot(143)
    plt.imshow(D, interpolation='none')
    plt.title("SSM")

    plt.subplot(144)
    #plt.imshow(v, aspect = 'auto')
    #plt.title("Eigenvectors")
    for k in range(n_clusters):
        plt.scatter(Y[idx == k, 0], Y[idx == k, 1])
    plt.show()
