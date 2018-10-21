import sys
import scipy.sparse as sparse
import scipy.stats
import scipy.sparse.linalg as slinalg
import numpy as np
import numpy.linalg as linalg
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter

def getUnweightedLaplacianEigsDense(A, neigs):
    DEG = scipy.sparse.dia_matrix((A.sum(1).flatten(), 0), A.shape)
    L = DEG.toarray() - A
    w, v = linalg.eigh(L)
    return (w[0:neigs], v[:, 0:neigs], L)

def getSymmetricLaplacianEigsDense(W, neigs):
    """
    Get eigenvectors of the weighted symmetric Laplacian
    Parameters
    ----------
    W: ndarray(N, N)
        A symmetric similarity matrix that has nonnegative entries everywhere
    neigs: int
        Number of eigenvectors to compute
    """
    D = scipy.sparse.dia_matrix((W.sum(1).flatten(), 0), W.shape).toarray()
    DInvSqrt = 1/np.sqrt(D)
    DInvSqrt[~np.isfinite(DInvSqrt)] = 1.0
    L = D - W
    LSym = DInvSqrt.dot(L.dot(DInvSqrt))
    w, v = linalg.eigh(LSym)
    return (w[0:neigs], v[:, 0:neigs], L)