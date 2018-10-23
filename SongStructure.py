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
from SongStructureGUI import saveResultsJSON

def plotFusionResults(Ws, vs, alllabels, times):
    """
    Show a plot of different adjacency matrices and their associated eigenvectors
    and cluster labels, if applicable
    Parameters
    ----------
    Ws: Dictionary of string:ndarray(N, N)
        Different adjacency matrix types
    vs: Dictionary of string:ndarray(N, k)
        Laplacian eigenvectors for different adjacency matrix types.
        If there is not a key for a particular adjacency matrix type, it isn't plotted
    alllabels: Dictionary of string:ndarray(N)
        Labels from spectral clustering for different adjacency matrix types.
        If there is not a key for a particular adjacency matrix type, it isn't plotted
    times: ndarray(N)
        A list of times corresponding to each row in Ws
    
    Returns
    -------
    fig: matplotlib.pyplot object
        Handle to the figure
    """
    fig = plt.figure(figsize=(10*len(Ws), 8))
    for i, name in enumerate(Ws):
        W = Ws[name]
        WShow = np.array(W)
        np.fill_diagonal(WShow, 0)
        plt.subplot2grid((1, 9*len(Ws)), (0, i*9), colspan=7)
        plt.pcolormesh(times, times, np.log(5e-2+WShow), cmap = 'afmhot')
        plt.gca().invert_yaxis()
        plt.title("%s Similarity Matrix"%name)
        plt.xlabel("Time (sec)")
        plt.ylabel("Time (sec)")
        if name in vs:
            v = vs[name]
            plt.subplot2grid((1, 9*len(Ws)), (0, i*9+7))
            plt.pcolormesh(np.arange(v.shape[1]+1), times, v, cmap='afmhot')
            plt.gca().invert_yaxis()
            plt.title("Laplacian")
            plt.xlabel("Eigenvector Num")
            plt.xticks(0.5 + np.arange(v.shape[1]), ["%i"%(i+1) for i in range(v.shape[1])])
        if name in alllabels:
            plt.subplot2grid((1, 9*len(Ws)), (0, i*9+8))
            levels = [-1] # Look at only finest level for now
            labels = np.zeros((W.shape[0], len(levels)))
            for k, level in enumerate(levels):
                labels[:, k] = alllabels[name][level]['labels']
            plt.pcolormesh(np.arange(labels.shape[1]+1), times, labels, cmap = 'tab20b')
            plt.gca().invert_yaxis()
            plt.title("Clusters")
    plt.tight_layout()
    return fig

def getFusedSimilarity(filename, sr, hop_length, win_fac, wins_per_block, K, reg_diag, reg_neighbs, niters, do_animation, plot_result):
    """
    Load in filename, compute features, average/stack delay, and do similarity
    network fusion (SNF) on all feature types
    Parameters
    ----------
    filename: string
        Path to music file
    sr: int
        Sample rate at which to sample file
    hop_length: int
        Hop size between frames in chroma and mfcc
    win_fac: int
        Number of frames to average (i.e. factor by which to downsample)
        If negative, then do beat tracking, and subdivide by |win_fac| times within each beat
    wins_per_block: int
        Number of aggregated windows per sliding window block
    K: int
        Number of nearest neighbors in SNF
    reg_diag: float 
        Regularization for self-similarity promotion
    reg_neighbs: float
        Regularization for direct neighbor similarity promotion
    niters: int
        Number of iterations in SNF
    do_animation: boolean
        Whether to plot and save images of the evolution of SNF
    plot_result: boolean
        Whether to plot the result of the fusion
    
    Returns
    -------
    {'Ws': An dictionary of weighted adjacency matrices for individual features
                    and the fused adjacency matrix, 
            'times': Time in seconds of each row in the similarity matrices} 
    """
    print("Loading %s..."%filename)
    y, sr = librosa.load(filename, sr=sr)
    
    if win_fac > 0:
        # Compute features in intervals evenly spaced by the hop size
        # but average within "win_fac" intervals of hop_length
        nHops = int((y.size-hop_length*win_fac*wins_per_block)/hop_length)
        intervals = np.arange(0, nHops, win_fac)
    else:
        # Compute features in intervals which are subdivided beats
        # by a factor of |win_fac|
        C = np.abs(librosa.cqt(y=y, sr=sr))
        _, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False, start_bpm=240)
        intervals = librosa.util.fix_frames(beats, x_max=C.shape[1])
        intervals = librosa.segment.subsegment(C, intervals, n_segments=abs(win_fac))


    # chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    # CQT chroma with 3x oversampling in pitch
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, bins_per_octave=12*3)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128, hop_length=hop_length)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=20)
    
    intervals = librosa.util.fix_frames(intervals, x_min=0, x_max=min(mfcc.shape[1], chroma.shape[1]))

    # chroma = librosa.util.sync(chroma, intervals)
    # median-aggregate chroma to suppress transients and passing tones
    chroma = librosa.util.sync(chroma, intervals, aggregate=np.median)
    mfcc = librosa.util.sync(mfcc, intervals)
    times = intervals*float(hop_length)/float(sr)


    n_frames = min(chroma.shape[1], mfcc.shape[1])
    chroma = chroma[:, :n_frames]
    mfcc = mfcc[:, :n_frames]

    #Do a delay embedding and compute SSMs
    XChroma = librosa.feature.stack_memory(chroma, n_steps=wins_per_block, mode='edge').T
    XMFCC = librosa.feature.stack_memory(mfcc, n_steps=wins_per_block, mode='edge').T
    DChroma = getCSMCosine(XChroma, XChroma) #Cosine distance
    DMFCC = getCSM(XMFCC, XMFCC) #Euclidean distance

    #Run similarity network fusion
    FeatureNames = ['MFCCs', 'Chromas']
    Ds = [DMFCC, DChroma]
    # Edge case: zeropad if it's too small
    for i, Di in enumerate(Ds):
        if Di.shape[0] < 2*K:
            D = np.zeros((2*K, 2*K))
            D[0:Di.shape[0], 0:Di.shape[1]] = Di
            Ds[i] = D

    (Ws, WFused) = doSimilarityFusion(Ds, K=K, niters=niters, \
        reg_diag=reg_diag, reg_neighbs=reg_neighbs, \
        do_animation=do_animation, PlotNames=FeatureNames, \
        PlotExtents=[times[0], times[-1]]) 
    WsDict = {}
    for n, W in zip(FeatureNames, Ws):
        WsDict[n] = W
    WsDict['Fused'] = WFused
    if plot_result:
        plotFusionResults(WsDict, {}, {}, times)
        plt.show()
    return {'Ws':WsDict, 'times':times}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True, help="Path to audio file")
    parser.add_argument('--do_animation', type=int, default=0, help='Save plots of iterations to disk')
    parser.add_argument('--plot_result', type=int, default=1, help='Plot the result of fusion')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate to use')
    parser.add_argument('--hop_length', type=int, default=512, help="Hop Size in samples")
    parser.add_argument('--win_fac', type=int, default=10, help="Number of windows to average in a frame.  If negative, then do beat tracking, and subdivide by |win_fac| times within each beat")
    parser.add_argument('--wins_per_block', type=int, default=20, help="Number of frames to stack in sliding window for every feature")
    parser.add_argument('--K', type=int, default=10, help="Number of nearest neighbors in similarity network fusion")
    parser.add_argument('--reg_diag', type=float, default=1.0, help="Regularization for self-similarity promotion")
    parser.add_argument('--reg_neighbs', type=float, default=0.5, help="Regularization for direct neighbor similarity promotion")
    parser.add_argument('--niters', type=int, default=10, help="Number of iterations in similarity network fusion")
    parser.add_argument('--neigs', type=int, default=8, help="Number of eigenvectors in the graph Laplacian")
    parser.add_argument('--matfilename', type=str, default="out.mat", help="Name of the .mat file to which to save the results")
    parser.add_argument('--jsonfilename', type=str, default="out.json", help="Name of the .json file to which to save results for viewing in the GUI")
    parser.add_argument('--diffusion_znormalize', type=int, default=0, help="Whether to perform Z-normalization with diffusion maps to spread things out more")


    opt = parser.parse_args()
    res = getFusedSimilarity(opt.filename, sr=opt.sr, \
        hop_length=opt.hop_length, win_fac=opt.win_fac, wins_per_block=opt.wins_per_block, \
        K=opt.K, reg_diag=opt.reg_diag, reg_neighbs=opt.reg_neighbs, niters=opt.niters, \
        do_animation=opt.do_animation, plot_result=opt.plot_result)
    sio.savemat(opt.matfilename, res)
    saveResultsJSON(opt.filename, res['times'], res['Ws'], opt.K, opt.neigs, opt.jsonfilename, opt.diffusion_znormalize)
