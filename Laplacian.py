import sys
import scipy
import scipy.sparse as sparse
import numpy as np
import numpy.linalg as linalg
import scipy.linalg as sclinalg

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
    _, v = linalg.eigh(L)
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
    _, v = linalg.eigh(LSym)
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
    _, v = sclinalg.eigh(L, D)
    return v