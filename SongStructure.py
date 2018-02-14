"""
Programmer: Chris Tralie
Purpose: To provide an interface for loading music, computing features, and
doing similarity fusion on those features to make a weighted adjacency matrix
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import librosa
import argparse
from CSMSSMTools import getCSM, getCSMCosine
from SimilarityFusion import doSimilarityFusion

import crema

ChordModel = crema.models.chord.ChordModel()

"""
TODO: Try SNF with different window lengths to better capture multiresolution structure
"""

def getFusedSimilarity(filename, sr = 22050, hopSize = 2048, winFac = 5, \
        winsPerBlock = 20, K = 10, NIters = 10, doAnimation = False):
    """
    Load in filename, compute features, average/stack delay, and do similarity
    network fusion
    :returns {'Ws': An array of weighted adjacency matrices for individual features, \
            'WFused': The fused adjacency matrix, \
            'times': Timestamps in seconds of each element in the adjacency matrices, \
            'BlockLen': Length of each stacked delay block in seconds, \
            'FeatureNames': Names of the features used, parallel with Ws}
    """
    print("Loading %s..."%filename)
    y, sr = librosa.load(filename, sr=sr)
#    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # square root the pitch predictions to make PPK embeddings
    chroma = ChordModel.outputs(y=y, sr=sr)['chord_pitch'].T**0.5
#    chroma = ChordModel.outputs(y=y, sr=sr)['chord_root'].T**0.5

    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128, hop_length=hopSize)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=20)
    
    #Compute features in intervals evenly spaced by the hop size
    #but average within "winFac" intervals of hopSize
    nHops = int((y.size-hopSize*winFac*winsPerBlock)/hopSize)
    intervals = np.arange(0, nHops, winFac)
    intervals = librosa.util.fix_frames(intervals, x_min=0, x_max=min(mfcc.shape[1], chroma.shape[1]))
    chroma = librosa.util.sync(chroma, intervals)
    mfcc = librosa.util.sync(mfcc, intervals)

    n_frames = min(chroma.shape[1], mfcc.shape[1])
    chroma = chroma[:, :n_frames]
    mfcc = mfcc[:, :n_frames]

    #Do a delay embedding and compute SSMs
    XChroma = librosa.feature.stack_memory(chroma, n_steps=winsPerBlock, mode='edge').T
    XMFCC = librosa.feature.stack_memory(mfcc, n_steps=winsPerBlock, mode='edge').T
    DChroma = getCSMCosine(XChroma, XChroma) #Cosine distance
    DMFCC = getCSM(XMFCC, XMFCC) #Euclidean distance

    #Run similarity network fusion
    FeatureNames = ['MFCCs', 'Chromas']
    Ds = [DMFCC, DChroma]
    interval = hopSize*winFac/float(sr)
    times = interval*np.arange(DMFCC.shape[0])
    print("Interval = %.3g Seconds, Block = %.3g Seconds"%(interval, interval*winsPerBlock))
    (Ws, WFused) = doSimilarityFusion(Ds, K = K, NIters = NIters, \
        regDiag = 1, regNeighbs=0.5, \
        doPlot = doAnimation, PlotNames = FeatureNames, \
        PlotExtents = [0, hopSize*winFac*float(DChroma.shape[0])/(sr)]) 
    
    return {'Ws':Ws, 'WFused':WFused, 'times':times, \
            'BlockLen':interval*winsPerBlock, 'FeatureNames':FeatureNames}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='MJBad.mp3', help="Path to audio file")
    parser.add_argument('--doanimation', type=int, default=0, help='Save plots of iterations to disk')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate to use')
    parser.add_argument('--hopSize', type=int, default=512, help="Hop Size in samples")
    parser.add_argument('--winFac', type=int, default=10, help="Number of windows to average in a frame")
    parser.add_argument('--winsPerBlock', type=int, default=20, help="Number of frames to stack in sliding window for every feature")
    parser.add_argument('--K', type=int, default=10, help="Number of nearest neighbors in similarity network fusion")
    parser.add_argument('--NIters', type=int, default=10, help="Number of iterations in similarity network fusion")
    parser.add_argument('--matfilename', type=str, default="out.mat", help="Name of the .mat file to which to save the results")


    opt = parser.parse_args()
    res = getFusedSimilarity(opt.filename, sr=opt.sr, \
        hopSize=opt.hopSize, winFac=opt.winFac, winsPerBlock=opt.winsPerBlock, \
        K=opt.K, NIters=opt.NIters, doAnimation=opt.doanimation)
    sio.savemat(opt.matfilename, res)
