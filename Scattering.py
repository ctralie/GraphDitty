"""
Code that wraps around Matlab ScatNet to compute scattering transforms
"""
import numpy as np
import scipy.io as sio
import scipy
import subprocess
import matplotlib.pyplot as plt
import os
import sys

def imresize(D, dims, kind='cubic', use_scipy=False):
    """
    Resize a floating point image
    Parameters
    ----------
    D : ndarray(M1, N1)
        Original image
    dims : tuple(M2, N2)
        The dimensions to which to resize
    kind : string
        The kind of interpolation to use
    use_scipy : boolean
        Fall back to scipy.misc.imresize.  This is a bad idea
        because it casts everything to uint8, but it's what I
        was doing accidentally for a while
    Returns
    -------
    D2 : ndarray(M2, N2)
        A resized array
    """
    if use_scipy:
        return scipy.misc.imresize(D, dims)
    else:
        M, N = dims
        x1 = np.array(0.5 + np.arange(D.shape[1]), dtype=np.float32)/D.shape[1]
        y1 = np.array(0.5 + np.arange(D.shape[0]), dtype=np.float32)/D.shape[0]
        x2 = np.array(0.5 + np.arange(N), dtype=np.float32)/N
        y2 = np.array(0.5 + np.arange(M), dtype=np.float32)/M
        f = scipy.interpolate.interp2d(x1, y1, D, kind=kind)
        return f(x2, y2)

def getPrefix():
    return 'scatnet-0.2'

def getScatteringTransform(imgs, renorm=True):
    intrenorm = 0
    if renorm:
        intrenorm = 1
    prefix = getPrefix()
    argimgs = np.zeros((imgs[0].shape[0], imgs[0].shape[1], len(imgs)))
    for i, img in enumerate(imgs):
        argimgs[:, :, i] = img
    sio.savemat("%s/x.mat"%prefix, {"x":argimgs, "renorm":intrenorm})
    subprocess.call(["matlab", "-nodisplay", "-nosplash", "-nodesktop", "-r", \
                    "cd %s; getScatteringImages; exit;"%(prefix)])
    res = sio.loadmat("%s/res.mat"%prefix)['res']
    images = []
    for i in range(len(res[0])):
        image = []
        for j in range(len(res[0][i][0])):
            image.append(res[0][i][0][j])
        images.append(image)
    return images

def flattenCoefficients(images):
    ret = []
    for im in images:
        ret.append(im.flatten())
    return np.array(ret)

def poolFeatures(image, res):
    M = int(image.shape[0]/res)
    N = int(image.shape[1]/res)
    ret = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            ret[i, j] = np.mean(image[i*res:(i+1)*res, j*res:(j+1)*res])
    return ret

def averageFeatures(image, res, k):
    """
    Resize features within each scattering path at a certain level
    Parameters
    ----------
    image: ndarray(M*res, N*res)
        An image holding MN scattering paths at a certain level, each
        with res*res coefficients
    res: int
        The resolution of each scattering path
    k: int
        The new downsampled resolution
    Returns
    -------
    imageavg: ndarray(M*k, N*k)
        The downsampled scattering coefficients
    """
    M = int(image.shape[0]/res)
    N = int(image.shape[1]/res)
    ret = np.zeros((M*k, N*k))
    for i in range(M):
        for j in range(N):
            ret[i*k:(i+1)*k, j*k:(j+1)*k] = imresize(image[i*res:(i+1)*res, j*res:(j+1)*res], (k, k))
    return ret

if __name__ == '__main__':
    D = np.random.randn(512, 512)
    res = getScatteringTransform([D])
    print(len(res[0]))
    for k in range(len(res[0])):
        print(res[0][k].shape)