import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import scipy.io as sio
from scipy import sparse
import time
from CSMSSMTools import *

def getDiffusionMap(K, neigs = 4, thresh=5e-4):
    """
    Perform diffusion maps with a unit timestep, automatically
    normalizing for nonuniform sampling
    Parameters
    ----------
    K: ndarray(N, N)
        A similarity kernel
    neigs: int
        Number of eigenvectors to compute
    thresh: float
        Threshold below which to zero out entries in
        the Markov chain approximation
    """
    tic = time.time()
    print("Building diffusion map matrix...")
    P = np.sum(K, 1)
    P[P == 0] = 1
    KHat = (K/P[:, None])/P[None, :]
    dRow = np.sum(KHat, 1)
    KHat[KHat < thresh] = 0
    KHat = sparse.csc_matrix(KHat)
    M = sparse.diags(dRow).tocsc()
    print("Elapsed Time: %.3g"%(time.time()-tic))
    print("Solving eigen system...")
    tic = time.time()
    # Solve a generalized eigenvalue problem
    w, v = sparse.linalg.eigsh(KHat, k=neigs, M=M, which='LM')
    print("Elapsed Time: %.3g"%(time.time()-tic))
    return w[None, :]*v

def getPinchedCircle(N):
    t = np.linspace(0, 2*np.pi, N+1)[0:N]
    x = np.zeros((N, 2))
    x[:, 0] = (1.5 + np.cos(2*t))*np.cos(t)
    x[:, 1] = (1.5 + np.cos(2*t))*np.sin(t)
    return x

def getTorusKnot(N, p, q):
    t = np.linspace(0, 2*np.pi, N+1)[0:N]
    X = np.zeros((N, 3))
    r = np.cos(q*t) + 2
    X[:, 0] = r*np.cos(p*t)
    X[:, 1] = r*np.sin(p*t)
    X[:, 2] = -np.sin(q*t)
    return X

def testDiffusionMaps():
    N = 400
    X = getPinchedCircle(N)
    tic = time.time()
    SSMOrig = getSSM(X)
    toc = time.time()
    print("Elapsed time SSM: ", toc - tic)
    Kappa = 0.1

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], 40, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
    plt.axis('equal')
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor((0.15, 0.15, 0.15))
    plt.title("Original Pinched Circle")
    plt.subplot(122)
    plt.imshow(SSMOrig, interpolation = 'nearest', cmap = 'afmhot')
    plt.title("Original SSM")
    plt.savefig("Diffusion0.svg", bbox_inches = 'tight')

    ts = [100]
    for t in ts:
        plt.clf()
        W = getW(SSMOrig, int(Kappa*SSMOrig.shape[0]))
        M = getDiffusionMap(W)
        SSM = getSSM(M)
        plt.subplot(121)
        X = M[:, [-2, -3]]
        plt.scatter(X[:, 0], X[:, 1], 40, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
        plt.title("2D Diffusion Map, t = %i, $\kappa = %g$"%(t, Kappa))
        plt.axis('equal')
        plt.xlim([np.min(X[:, 0]) - 0.001, np.max(X[:, 0]) + 0.001])
        plt.ylim([np.min(X[:, 1]) - 0.001, np.max(X[:, 1]) + 0.001])
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor((0.15, 0.15, 0.15))
        plt.subplot(122)
        plt.imshow(SSM, interpolation = 'nearest', cmap = 'afmhot')
        plt.title("Diffusion Distance")
        plt.savefig("Diffusion%i.svg"%t, bbox_inches = 'tight')

if __name__ == '__main__':
    testDiffusionMaps()