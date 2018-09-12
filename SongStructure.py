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
from Laplacian import getLaplacianEigsDense
from SongStructureGUI import saveResultsJSON

def getFusedSimilarity(filename, sr, hop_length, win_fac, wins_per_block, K, reg_diag, reg_neighbs, niters, neigs, do_animation, plot_result):
    """
    Load in filename, compute features, average/stack delay, and do similarity
    network fusion (SNF)
    :param filename: Path to music file
    :param sr: Sample rate at which to sample file
    :param hop_length: Hop size between frames in chroma and mfcc
    :param win_fac: Number of frames to average (i.e. factor by which to downsample)
    :param wins_per_block: Number of aggregated windows per sliding window block
    :param K: Number of nearest neighbors in SNF
    :param reg_diag: Regularization for self-similarity promotion
    :param reg_neighbs: Regularization for direct neighbor similarity promotion
    :param niters: Number of iterations in SNF
    :param neigs: Number of eigenvectors in the Laplacian
    :param do_animation: Whether to plot and save images of the evolution of SNF
    :param plot_result: Whether to plot the result of the fusion
    :returns {'Ws': An array of weighted adjacency matrices for individual features, \
            'WFused': The fused adjacency matrix, \
            'times': Timestamps in seconds of each element in the adjacency matrices, \
            'BlockLen': Length of each stacked delay block in seconds, \
            'FeatureNames': Names of the features used, parallel with Ws}
    """
    print("Loading %s..."%filename)
    y, sr = librosa.load(filename, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128, hop_length=hop_length)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=20)
    
    #Compute features in intervals evenly spaced by the hop size
    #but average within "win_fac" intervals of hop_length
    nHops = int((y.size-hop_length*win_fac*wins_per_block)/hop_length)
    intervals = np.arange(0, nHops, win_fac)
    intervals = librosa.util.fix_frames(intervals, x_min=0, x_max=min(mfcc.shape[1], chroma.shape[1]))
    chroma = librosa.util.sync(chroma, intervals)
    mfcc = librosa.util.sync(mfcc, intervals)

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
    time_interval = hop_length*win_fac/float(sr)
    print("Interval = %.3g Seconds, Block = %.3g Seconds"%(time_interval, time_interval*wins_per_block))
    PlotExtents = [0, time_interval*DMFCC.shape[0]]
    (Ws, WFused) = doSimilarityFusion(Ds, K=K, niters=niters, \
        reg_diag=reg_diag, reg_neighbs=reg_neighbs, \
        do_animation=do_animation, PlotNames=FeatureNames, \
        PlotExtents=PlotExtents) 
    (w, v, L) = getLaplacianEigsDense(WFused, neigs)
    v = v[:, 1::]
    if plot_result:
        plt.figure(figsize=(12, 10))
        WShow = np.array(WFused)
        np.fill_diagonal(WShow, 0)
        plt.subplot2grid((1, 8), (0, 0), colspan=7)
        plt.imshow(np.log(5e-2+WShow), interpolation = 'nearest', cmap = 'afmhot', \
        extent = (PlotExtents[0], PlotExtents[1], PlotExtents[1], PlotExtents[0]))
        plt.title("Similarity Matrix")
        plt.xlabel("Time (sec)")
        plt.ylabel("Time (sec)")
        plt.subplot2grid((1, 8), (0, 7))
        plt.imshow(v, cmap='afmhot', interpolation = 'nearest', aspect='auto', \
            extent=(0, v.shape[1], PlotExtents[1], PlotExtents[0]))
        plt.title("Laplacian")
        plt.xlabel("Eigenvector Num")
        plt.xticks(0.5 + np.arange(v.shape[1]), ["%i"%(i+1) for i in range(v.shape[1])])
        plt.show()
    return {'Ws':Ws, 'WFused':WFused, 'v':v, 'time_interval':time_interval,\
            'BlockLen':time_interval*wins_per_block, 'FeatureNames':FeatureNames}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True, help="Path to audio file")
    parser.add_argument('--do_animation', type=int, default=0, help='Save plots of iterations to disk')
    parser.add_argument('--plot_result', type=int, default=1, help='Plot the result of fusion')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate to use')
    parser.add_argument('--hop_length', type=int, default=512, help="Hop Size in samples")
    parser.add_argument('--win_fac', type=int, default=10, help="Number of windows to average in a frame")
    parser.add_argument('--wins_per_block', type=int, default=20, help="Number of frames to stack in sliding window for every feature")
    parser.add_argument('--K', type=int, default=10, help="Number of nearest neighbors in similarity network fusion")
    parser.add_argument('--reg_diag', type=float, default=1.0, help="Regularization for self-similarity promotion")
    parser.add_argument('--reg_neighbs', type=float, default=0.5, help="Regularization for direct neighbor similarity promotion")
    parser.add_argument('--niters', type=int, default=10, help="Number of iterations in similarity network fusion")
    parser.add_argument('--neigs', type=int, default=8, help="Number of eigenvectors in the graph Laplacian")
    parser.add_argument('--matfilename', type=str, default="out.mat", help="Name of the .mat file to which to save the results")
    parser.add_argument('--jsonfilename', type=str, default="out.json", help="Name of the .json file to which to save results for viewing in the GUI")


    opt = parser.parse_args()
    res = getFusedSimilarity(opt.filename, sr=opt.sr, \
        hop_length=opt.hop_length, win_fac=opt.win_fac, wins_per_block=opt.wins_per_block, \
        K=opt.K, reg_diag=opt.reg_diag, reg_neighbs=opt.reg_neighbs, niters=opt.niters, \
        neigs=opt.neigs, do_animation=opt.do_animation, plot_result=opt.plot_result)
    sio.savemat(opt.matfilename, res)
    saveResultsJSON(opt.filename, res['time_interval'], res['WFused'], opt.K, res['v'], opt.jsonfilename)
    
