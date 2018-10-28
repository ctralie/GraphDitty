"""
Programmer: Chris Tralie
Purpose: To provide an interface for loading music, computing features, and
doing similarity fusion on those features to make a weighted adjacency matrix
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.interpolate
import os
import librosa
import librosa.display
import argparse
from CSMSSMTools import getCSM, getCSMCosine
from SimilarityFusion import doSimilarityFusion
from SongStructureGUI import saveResultsJSON
import subprocess

MANUAL_AUDIO_LOAD = True
FFMPEG_BINARY = "ffmpeg"

def plotFusionResults(Ws, vs, alllabels, times, win_fac, intervals_hier = [], labels_hier = []):
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
    win_fac: int
        Number of frames that have been averaged in each window
        If negative, beat tracking has been done, and the intervals are possibly non-uniform
        This means that a mesh plot will be necessary
    Returns
    -------
    fig: matplotlib.pyplot object
        Handle to the figure
    """
    nrows = int(np.ceil(len(Ws)/3.0))
    fig = plt.figure(figsize=(0.5*32, 0.5*8*nrows))
    time_uniform = win_fac >= 0
    for i, name in enumerate(Ws):
        W = Ws[name]
        WShow = np.log(5e-2+W)
        np.fill_diagonal(WShow, 0)
        row, col = np.unravel_index(i, (nrows, 3))
        plt.subplot2grid((nrows, 8*3), (row, col*8), colspan=7)
        if time_uniform:
            plt.imshow(WShow, cmap ='afmhot', extent=(times[0], times[-1], times[-1], times[0]), interpolation='nearest')
        else:
            plt.pcolormesh(times, times, WShow, cmap = 'afmhot')
            plt.gca().invert_yaxis()
        plt.title("%s Similarity Matrix"%name)
        if row == nrows-1:
            plt.xlabel("Time (sec)")
        if col == 0:
            plt.ylabel("Time (sec)")
        if name in alllabels:
            plt.subplot2grid((nrows, 8*3), (row, col*8+7))
            levels = [0] # Look at only finest level for now
            labels = np.zeros((W.shape[0], len(levels)))
            for k, level in enumerate(levels):
                labels[:, k] = alllabels[name][level]['labels']
            if time_uniform:
                plt.imshow(labels, cmap = 'tab20b', interpolation='nearest', aspect='auto', extent=(0, 1, times[-1], times[0]))
            else:
                plt.pcolormesh(np.arange(labels.shape[1]+1), times, labels, cmap = 'tab20b')
                plt.gca().invert_yaxis()
            plt.axis('off')
            plt.title("Clusters")
    #plt.tight_layout()
    if len(labels_hier) > 0:
        for k in range(2):
            plt.subplot2grid((nrows, 8*3), (nrows-1, 10+k*3))
            labels = []
            labelsdict = {}
            for a in labels_hier[k]:
                if not a in labelsdict:
                    labelsdict[a] = len(labelsdict)
                labels.append(labelsdict[a])
            labels = np.array(labels)
            plt.pcolormesh(np.arange(2), intervals_hier[k][:, 0], np.concatenate((labels[:, None], labels[:, None]), 1), cmap='tab20b')
            for i in range(intervals_hier[k].shape[0]):
                t = intervals_hier[k][i, 0]
                plt.plot([0, 1], [t, t], 'k', linestyle='--')
            plt.gca().invert_yaxis()
    return fig

def getFusedSimilarity(filename, sr, hop_length, win_fac, wins_per_block, K, reg_diag, reg_neighbs, niters, do_animation, plot_result, do_crema=True):
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
        Number of nearest neighbors in SNF.  If -1, then autotuned to sqrt(N)
        for an NxN similarity matrix
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
    do_crema: boolean
        Whether to include precomputed crema in the fusion
    Returns
    -------
    {'Ws': An dictionary of weighted adjacency matrices for individual features
                    and the fused adjacency matrix, 
            'times': Time in seconds of each row in the similarity matrices,
            'K': The number of nearest neighbors actually used} 
    """
    ## Step 1: Load audio
    print("Loading %s..."%filename)
    if MANUAL_AUDIO_LOAD:
        subprocess.call([FFMPEG_BINARY, "-i", filename, "-ar", "%i"%sr, "-ac", "1", "%s.wav"%filename])
        sr, y = sio.wavfile.read("%s.wav"%filename)
        y = y/2.0**15
        os.remove("%s.wav"%filename)
    else:
        y, sr = librosa.load(filename, sr=sr)
    
    ## Step 2: Figure out intervals to which to sync features
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

    ## Step 3: Compute features
    # 1) CQT chroma with 3x oversampling in pitch
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, bins_per_octave=12*3)

    # 2) Exponentially liftered MFCCs
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128, hop_length=hop_length)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=20)
    lifterexp = 0.6
    coeffs = np.arange(mfcc.shape[0])**lifterexp
    coeffs[0] = 1
    mfcc = coeffs[:, None]*mfcc

    # 3) Tempograms
    #  Use a super-flux max smoothing of 5 frequency bands in the oenv calculation
    SUPERFLUX_SIZE = 5
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length,
                                        max_size=SUPERFLUX_SIZE)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)

    # 4) Crema
    if do_crema:
        matfilename = "%s_crema.mat"%filename
        if not os.path.exists(matfilename):
            print("****WARNING: PRECOMPUTED CREMA DOES NOT EXIST****")
            do_crema = False
        else:
            data = sio.loadmat(matfilename)
            fac = (float(sr)/44100.0)*4096.0/hop_length
            times_orig = fac*np.arange(len(data['chord_bass']))
            times_new = np.arange(mfcc.shape[1])
            interp = scipy.interpolate.interp1d(times_orig, data['chord_pitch'].T, kind='nearest', fill_value='extrapolate')
            chord_pitch = interp(times_new)
    
    ## Step 4: Synchronize features to intervals
    n_frames = np.min([chroma.shape[1], mfcc.shape[1], tempogram.shape[1]])
    if do_crema:
        n_frames = min(n_frames, chord_pitch.shape[1])
    # median-aggregate chroma to suppress transients and passing tones
    intervals = librosa.util.fix_frames(intervals, x_min=0, x_max=n_frames)
    times = intervals*float(hop_length)/float(sr)

    chroma = librosa.util.sync(chroma, intervals, aggregate=np.median)
    chroma = chroma[:, :n_frames]
    mfcc = librosa.util.sync(mfcc, intervals)
    mfcc = mfcc[:, :n_frames]
    tempogram = librosa.util.sync(tempogram, intervals)
    tempogram = tempogram[:, :n_frames]
    if do_crema:
        chord_pitch = librosa.util.sync(chord_pitch, intervals)
        chord_pitch = chord_pitch[:, :n_frames]
    

    ## Step 5: Do a delay embedding and compute SSMs
    XChroma = librosa.feature.stack_memory(chroma, n_steps=wins_per_block, mode='edge').T
    DChroma = getCSMCosine(XChroma, XChroma) #Cosine distance
    XMFCC = librosa.feature.stack_memory(mfcc, n_steps=wins_per_block, mode='edge').T
    DMFCC = getCSM(XMFCC, XMFCC) #Euclidean distance
    XTempogram = librosa.feature.stack_memory(tempogram, n_steps=wins_per_block, mode='edge').T
    DTempogram = getCSM(XTempogram, XTempogram)
    if do_crema:
        XChordPitch = librosa.feature.stack_memory(chord_pitch, n_steps=wins_per_block, mode='edge').T
        DChordPitch = getCSMCosine(XChordPitch, XChordPitch)

    ## Step 5: Run similarity network fusion
    FeatureNames = ['MFCCs', 'Chromas', 'Tempogram']
    Ds = [DMFCC, DChroma, DTempogram]
    if do_crema:
        FeatureNames.append('Crema')
        Ds.append(DChordPitch)
    # Edge case: If it's too small, zeropad SSMs
    for i, Di in enumerate(Ds):
        if Di.shape[0] < 2*K:
            D = np.zeros((2*K, 2*K))
            D[0:Di.shape[0], 0:Di.shape[1]] = Di
            Ds[i] = D
    pK = K
    if K == -1:
        pK = int(np.round(2*np.log(Ds[0].shape[0])/np.log(2)))
        print("Autotuned K = %i"%pK)
    # Do fusion on all features
    (Ws, WFused) = doSimilarityFusion(Ds, K=pK, niters=niters, \
        reg_diag=reg_diag, reg_neighbs=reg_neighbs, \
        do_animation=do_animation, PlotNames=FeatureNames, \
        PlotExtents=[times[0], times[-1]]) 
    WsDict = {}
    for n, W in zip(FeatureNames, Ws):
        WsDict[n] = W
    WsDict['Fused'] = WFused
    # Do fusion with only Chroma and MFCC
    (_, WsDict['Fused MFCC/Chroma']) = doSimilarityFusion([DMFCC, DChroma], K=pK, niters=niters, \
        reg_diag=reg_diag, reg_neighbs=reg_neighbs)
    if do_crema:
        # Do fusion with tempograms and Crema if Crema is available
        (_, WsDict['Fused Tgram/Crema']) = doSimilarityFusion([DTempogram, DChordPitch], K=pK, niters=niters, \
            reg_diag=reg_diag, reg_neighbs=reg_neighbs)
    if plot_result:
        plotFusionResults(WsDict, {}, {}, times, win_fac)
        plt.savefig("%s_Plot.png"%filename, bbox_inches='tight')
    return {'Ws':WsDict, 'times':times, 'K':pK}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True, help="Path to audio file")
    parser.add_argument('--do_animation', type=int, default=0, help='Save plots of iterations to disk')
    parser.add_argument('--plot_result', type=int, default=1, help='Plot the result of fusion')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate to use')
    parser.add_argument('--hop_length', type=int, default=512, help="Hop Size in samples")
    parser.add_argument('--win_fac', type=int, default=10, help="Number of windows to average in a frame.  If negative, then do beat tracking, and subdivide by |win_fac| times within each beat")
    parser.add_argument('--wins_per_block', type=int, default=20, help="Number of frames to stack in sliding window for every feature")
    parser.add_argument('--K', type=int, default=10, help="Number of nearest neighbors in similarity network fusion.  If -1, then autotune to sqrt(N) for an NxN similarity matrix")
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
        do_animation=opt.do_animation, plot_result=opt.plot_result, do_crema=False)
    sio.savemat(opt.matfilename, res)
    saveResultsJSON(opt.filename, res['times'], res['Ws'], res['K'], opt.neigs, opt.jsonfilename, opt.diffusion_znormalize)
