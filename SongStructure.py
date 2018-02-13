import numpy as np
import matplotlib.pyplot as plt
import librosa
from CSMSSMTools import *
from SimilarityFusion import *

"""
TODO: Try SNF with different window lengths to capture multiresolution structure

"""

def getFeatures(filename, Fs = 22050, hopSize = 512, winFac = 5, winsPerBlock = 20):
    print("Loading %s..."%filename)
    y, sr = librosa.load(filename, sr=Fs)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=20)

    #Compute features in intervals evenly spaced by the hop size
    #but average within "winFac" intervals of hopSize
    nHops = int((y.size-hopSize*winFac*winsPerBlock)/hopSize)
    intervals = np.arange(0, nHops, winFac)
    intervals = librosa.util.fix_frames(intervals, x_min=0, x_max=chroma.shape[1])
    chroma = librosa.util.sync(chroma, intervals)
    mfcc = librosa.util.sync(mfcc, intervals)

    #Do a delay embedding and compute SSMs
    XChroma = librosa.feature.stack_memory(chroma, n_steps=winsPerBlock, mode='edge').T
    XMFCC = librosa.feature.stack_memory(mfcc, n_steps=winsPerBlock, mode='edge').T
    DChroma = getCSMCosine(XChroma, XChroma) #Cosine distance
    DMFCC = getCSM(XMFCC, XMFCC) #Euclidean distance

    Ds = [DMFCC, DChroma]
    print("Interval = %.3g Seconds"%(hopSize*winFac/float(sr)))
    D = doSimilarityFusion(Ds, K = 10, NIters = 20, \
        regDiag = 1, regNeighbs=0.5, \
        PlotNames = ['MFCCs', 'Chromas'], \
        PlotExtents = [0, hopSize*winFac*float(DChroma.shape[0])/(sr)]) 

if __name__ == '__main__':
    hopSize = 512
    winFac = 5
    winsPerBlock = 20
    filename = "MJBad.mp3"
    Fs = 22050
    getFeatures(filename, Fs=Fs, hopSize=512, winFac=winFac, winsPerBlock = winsPerBlock)