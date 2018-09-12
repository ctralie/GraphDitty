#Programmer: Chris Tralie
#Purpose: To extract similarity alignments for use in the GUI
import numpy as np
import os
import scipy.misc
import matplotlib.pyplot as plt
import json
import base64
from SimilarityFusion import *

def imresize(D, dims, kind='cubic'):
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
    Returns
    -------
    D2 : ndarray(M2, N2)
        A resized array
    """
    M, N = dims
    x1 = np.array(0.5 + np.arange(D.shape[1]), dtype=np.float32)/D.shape[1]
    y1 = np.array(0.5 + np.arange(D.shape[0]), dtype=np.float32)/D.shape[0]
    x2 = np.array(0.5 + np.arange(N), dtype=np.float32)/N
    y2 = np.array(0.5 + np.arange(M), dtype=np.float32)/M
    f = scipy.interpolate.interp2d(x1, y1, D, kind=kind)
    return f(x2, y2)

def getBase64File(filename):
    fin = open(filename, "rb")
    b = fin.read()
    b = base64.b64encode(b)
    fin.close()
    return b.decode("ASCII")

def getBase64PNGImage(pD, cmapstr, logfloor = 0):
    """
    Get an image as a base64 string
    """
    D = np.array(pD)
    if logfloor > 0:
        D = np.log(D + logfloor)
    c = plt.get_cmap(cmapstr)
    D = D-np.min(D)
    D = np.round(255.0*D/np.max(D))
    C = c(np.array(D, dtype=np.int32))
    scipy.misc.imsave("temp.png", C)
    b = getBase64File("temp.png")
    os.remove("temp.png")
    return "data:image/png;base64, " + b

#http://stackoverflow.com/questions/1447287/format-floats-with-standard-json-module
class PrettyFloat(float):
    def __repr__(self):
        return '%.4g' % self
def pretty_floats(obj):
    if isinstance(obj, float):
        return PrettyFloat(obj)
    elif isinstance(obj, dict):
        return dict((k, pretty_floats(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return map(pretty_floats, obj)
    return obj

def get_graph_obj(W, K, res = 400):
    """
    Return an object corresponding to a nearest neighbor graph
    Parameters
    ----------
    W: ndarray(N, N)
        The N x N similarity matrix, where each row is time_interval
        apart from its adjacent rows
    K: int
        Number of nearest neighbors to use in graph representation
    res: int
        Target resolution of resized image
    """
    fac = 1
    if res > -1:
        fac = int(np.round(W.shape[0]/float(res)))
        res = W.shape[0]/fac
        WRes = imresize(W, (res, res))
    else:
        res = W.shape[0]
        WRes = np.array(W)
    np.fill_diagonal(WRes, 0)
    pix = np.arange(res)
    I, J = np.meshgrid(pix, pix)
    WRes[np.abs(I - J) == 1] = np.max(WRes)
    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255,res)), dtype=np.int32))
    C = np.array(np.round(C[:, 0:3]*255), dtype=int)
    colors = C.tolist()

    K = min(int(np.round(K*2.0/fac)), res) # Use slightly more edges
    print("res = %i, K = %i"%(res, K))
    S = getS(WRes, K).tocoo()
    I, J, V = S.row, S.col, S.data
    V *= 10
    #eps = 1e-5
    #V[V < eps] = eps
    #V = np.log(V/eps)
    ret = {}
    ret["nodes"] = [{"id":"%i"%i, "color":colors[i]} for i in range(res)]
    ret["links"] = [{"source":"%i"%I[i], "target":"%i"%J[i], "value":"%.3g"%V[i]} for i in range(I.shape[0])]
    ret["fac"] = fac
    return ret

def saveResultsJSON(filename, time_interval, W, K, v, jsonfilename):
    """
    Save a JSON file holding the audio and structure information, which can 
    be parsed by SongStructureGUI.html.  Audio and images are stored as
    base64 for simplicity
    
    Parameters
    ----------
    filename: string
        Path to audio
    time_interval: float
        Time interval between adjacent windows
    W: ndarray(N, N)
        The N x N similarity matrix, where each row is time_interval
        apart from its adjacent rows
    K: int
        Number of nearest neighbors to use in graph representation
    v: ndarray(N, k)
        Eigenvectors of weighted Laplacian
    jsonfilename: string
        File to which to save the .json file
    """
    Results = {'songname':filename, 'time_interval':time_interval}
    print("Saving results...")
    #Add music as base64 files
    path, ext = os.path.splitext(filename)
    Results['audio'] = "data:audio/%s;base64, "%ext[1::] + getBase64File(filename)
    WOut = np.array(W)
    np.fill_diagonal(WOut, 0)
    Results['W'] = getBase64PNGImage(WOut, 'afmhot', 5e-2)
    Results['dim'] = W.shape[0]
    
    # Resize the eigenvectors so they're easier to see
    fac = 10
    vout = np.zeros((v.shape[1]*fac, v.shape[0]))
    for i in range(fac):
        vout[i::fac, :] = v.T
    Results['v'] = getBase64PNGImage(vout, 'afmhot')
    Results['v_height'] = vout.shape[0]

    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255,W.shape[0])), dtype=np.int32))
    C = np.array(np.round(C[:, 0:3]*255), dtype=int)
    Results['colors'] = C.tolist()
    Results['graph'] = json.dumps(get_graph_obj(WOut, K))

    fout = open(jsonfilename, "w")
    fout.write(json.dumps(Results))
    fout.close()

if __name__ == '__main__':
    filename = "MJ.mp3"
    path, ext = os.path.splitext(filename)
    res = "data:audio/%s;base64, "%ext[1::] + getBase64File(filename)
    print(res)