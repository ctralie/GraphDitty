import sys
import scipy
import scipy.sparse as sparse
import numpy as np
import numpy.linalg as linalg
import scipy.linalg as sclinalg
from sklearn.cluster import KMeans

EVEC_SMOOTH = 9

def getUnweightedLaplacianEigsDense(W):
    """
    Get eigenvectors of the unweighted Laplacian
    Parameters
    ----------
    W: ndarray(N, N)
        A symmetric similarity matrix that has nonnegative entries everywhere
    
    Returns
    -------
    v: ndarray(N, N)
        A matrix of eigenvectors
    """
    D = scipy.sparse.dia_matrix((W.sum(1).flatten(), 0), W.shape).toarray()
    L = D - W
    try:
        _, v = linalg.eigh(L)
    except:
        return np.zeros_like(W)
    return v

def getSymmetricLaplacianEigsDense(W):
    """
    Get eigenvectors of the weighted symmetric Laplacian
    Parameters
    ----------
    W: ndarray(N, N)
        A symmetric similarity matrix that has nonnegative entries everywhere
    
    Returns
    -------
    v: ndarray(N, N)
        A matrix of eigenvectors
    """
    D = scipy.sparse.dia_matrix((W.sum(1).flatten(), 0), W.shape).toarray()
    L = D - W
    SqrtD = np.sqrt(D)
    SqrtD[SqrtD == 0] = 1.0
    DInvSqrt = 1/SqrtD
    LSym = DInvSqrt.dot(L.dot(DInvSqrt))
    try:
        _, v = linalg.eigh(LSym)
    except:
        return np.zeros_like(W)
    return v

def getRandomWalkLaplacianEigsDense(W):
    """
    Get eigenvectors of the random walk Laplacian by solving
    the generalized eigenvalue problem
    L*u = lam*D*u
    Parameters
    ----------
    W: ndarray(N, N)
        A symmetric similarity matrix that has nonnegative entries everywhere
    
    Returns
    -------
    v: ndarray(N, N)
        A matrix of eigenvectors
    """
    D = scipy.sparse.dia_matrix((W.sum(1).flatten(), 0), W.shape).toarray()
    L = D - W
    try:
        _, v = sclinalg.eigh(L, D)
    except:
        return np.zeros_like(W)
    return v

def spectralClusterSequential(v, dim, times, rownorm=False):
    """
    Given Laplacian eigenvectors associated with a time series, perform 
    spectral clustering, and return a compressed representation of
    the clusters, merging adjacent points with the same label into one cluster
    Parameters
    ----------
    v: ndarray(N, k)
        A matrix of eigenvectors, excluding the zeroeth
    dim: int
        Dimension of spectral clustering, <= k
    times: ndarray(N)
        Time in seconds of each row of the eigenvector matrix
    rownorm: boolean
        Whether to normalize each row (if using symmetric Laplacian)

    Returns
    -------
    labels: ndarray(N)
        Cluster membership for each point
    intervals_hier: ndarray (N, 2)
        Intervals (in seconds) of annotations
    labels_hier: list(strings)
        Corresponding segment labels for annotations
    """
    assert dim <= v.shape[1]
    x = np.array(v[:, 0:dim])
    if EVEC_SMOOTH > 0:
        x = scipy.ndimage.median_filter(x, size=(EVEC_SMOOTH, 1))
    if rownorm:
        norms = np.sqrt(np.sum(x**2, 1))
        norms[norms == 0] = 1
        x /= norms[:, None]
    labels = KMeans(n_clusters = dim, n_init=50, max_iter=500).fit(x).labels_
    splits = np.where(np.abs(labels[1::]-labels[0:-1]) > 0)[0]+1
    splits = np.concatenate(([0], splits, [labels.size]))
    # Handle edge case with small audio
    splits[splits >= times.size] = times.size
    if np.sum(splits == times.size) > 1:
        splits = np.unique(splits)
    groups = np.split(labels, splits)[1:-1]
    intervals_hier = np.zeros((len(groups), 2))
    timesext = np.array(times.tolist() + [times[-1]])
    intervals_hier[:, 0] = timesext[splits[0:-1]]
    intervals_hier[:, 1] = timesext[splits[1::]]
    x = np.zeros((intervals_hier.shape[0], 3))
    labels_hier = ['%i'%g[0] for g in groups]
    return {'labels':labels, 'intervals_hier':intervals_hier, 'labels_hier':labels_hier}