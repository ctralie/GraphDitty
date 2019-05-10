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
from SimilarityFusion import doSimilarityFusionWs, getW
from SongStructureGUI import saveResultsJSON
import subprocess

REC_SMOOTH = 9
MANUAL_AUDIO_LOAD = True
FFMPEG_BINARY = "ffmpeg"

def plotFusionResults(Ws, vs, alllabels, times, win_fac, wins_per_block = 1, intervals_hier = [], labels_hier = []):
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
    wins_per_block: int
    Returns
    -------
    fig: matplotlib.pyplot object
        Handle to the figure
    """
    nrows = int(np.ceil(len(Ws)/3.0))
    fac = 0.5
    if len(intervals_hier) > 0:
        fig = plt.figure(figsize=(fac*32, fac*8*nrows))
    else:
        fig = plt.figure(figsize=(fac*24, fac*8*nrows))
    time_uniform = win_fac >= 0
    for i, name in enumerate(['Chromas', 'MFCCs', 'Fused MFCC/Chroma', 'CREMA', 'Tempogram', 'Fused']):
        W = Ws[name]
        floor = np.quantile(W.flatten(), 0.01)
        WShow = np.log(W+floor)
        np.fill_diagonal(WShow, 0)
        row, col = np.unravel_index(i, (nrows, 3))
        if len(intervals_hier) > 0:
            plt.subplot2grid((nrows, 8*3), (row, col*8), colspan=7)
        else:
            plt.subplot(nrows, 3, i+1)
        if time_uniform:
            plt.imshow(WShow, cmap ='magma_r', extent=(times[0], times[-1], times[-1], times[0]), interpolation='nearest')
        else:
            plt.pcolormesh(times, times, WShow, cmap = 'magma_r')
            plt.gca().invert_yaxis()
        if 'Fused' in name:
            plt.title(name)
        else:
            plt.title("%s lags=%i"%(name, wins_per_block))
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
    plt.tight_layout()
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

def getFusedSimilarity(filename, sr, hop_length, win_fac, wins_per_block, K, \
                        reg_diag, reg_neighbs, niters, do_animation, plot_result, \
                        do_mfcc = True, do_chroma = True, do_tempogram = True, \
                        do_crema=True, precomputed_crema = True):
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
    do_mfcc: boolean
        Compute MFCC in the fusion
    do_chroma: boolean
        Include chroma in the fusion
    do_tempogram: boolean
        Include tempogram in the fusion
    do_crema: boolean
        Include CREMA in the fusion
    precomputed_crema: boolean
        If doing CREMA, also use precomputed CREMA in the file "$filename_crema.mat"
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
    n_frames = np.inf
    features = [] #A list of tuples (feature array, name string, aggregation function, distance function)
    # 1) CQT chroma with 3x oversampling in pitch
    if do_chroma:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, bins_per_octave=12*3)
        n_frames = min(n_frames, chroma.shape[1])
        # median-aggregate chroma to suppress transients and passing tones
        features.append({"x":chroma, "name":"Chromas", "agg_fn":np.median, "dist_fn":getCSMCosine})

    # 2) Exponentially liftered MFCCs
    if do_mfcc:
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128, hop_length=hop_length)
        log_S = librosa.power_to_db(S, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=20)
        lifterexp = 0.6
        coeffs = np.arange(mfcc.shape[0])**lifterexp
        coeffs[0] = 1
        mfcc = coeffs[:, None]*mfcc
        n_frames = min(n_frames, mfcc.shape[1])
        features.append({"x":mfcc, "name":"MFCCs", "agg_fn":np.mean, "dist_fn":getCSM})

    # 3) Tempograms
    if do_tempogram:
        #  Use a super-flux max smoothing of 5 frequency bands in the oenv calculation
        SUPERFLUX_SIZE = 5
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length,
                                            max_size=SUPERFLUX_SIZE)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
        features.append({"x":tempogram, "name":"Tempogram", "agg_fn":np.mean, "dist_fn":getCSM})

    # 4) Crema
    if do_crema:
        print("DOING CREMA")
        if precomputed_crema:
            matfilename = "%s_crema.mat"%filename
            if os.path.exists(matfilename):
                data = sio.loadmat(matfilename)
            else:
                print("WARNING: Want to use precomputed CREMA but file %s doesn't exist"%matfilename)
                do_crema = False
        else:
            import crema
            model = crema.models.chord.ChordModel()
            y44100, sr = librosa.load(filename, sr=44100)
            data = model.outputs(y=y44100, sr=44100)
        if do_crema:
            fac = (float(sr)/44100.0)*4096.0/hop_length
            times_orig = fac*np.arange(len(data['chord_bass']))
            times_new = np.arange(mfcc.shape[1])
            interp = scipy.interpolate.interp1d(times_orig, data['chord_pitch'].T, kind='nearest', fill_value='extrapolate')
            chord_pitch = interp(times_new)
            features.append({"x":chord_pitch, "name":"CREMA", "agg_fn":np.median, "dist_fn":getCSMCosine})
    
    ## Step 4: Synchronize features to intervals, do delay embedding, compute SSM
    intervals = librosa.util.fix_frames(intervals, x_min=0, x_max=n_frames)
    times = intervals*float(hop_length)/float(sr)
    FeatureNames = []
    Ds = []
    for i, f in enumerate(features):
        x = librosa.util.sync(f['x'], intervals, aggregate=f['agg_fn'])[:, :n_frames]
        X = librosa.feature.stack_memory(x, n_steps=wins_per_block, mode='edge').T
        Ds.append(f['dist_fn'](X, X))
        FeatureNames.append(f['name'])
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
    Ws = [getW(D, pK) for D in Ds]
    if REC_SMOOTH > 0:
        df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
        Ws = [df(W, size=(1, REC_SMOOTH)) for W in Ws]

    WFused = doSimilarityFusionWs(Ws, K=pK, niters=niters, \
        reg_diag=reg_diag, reg_neighbs=reg_neighbs, \
        do_animation=do_animation, PlotNames=FeatureNames, \
        PlotExtents=[times[0], times[-1]]) 
    WsDict = {}
    for n, W in zip(FeatureNames, Ws):
        WsDict[n] = W
    WsDict['Fused'] = WFused
    WFused2 = doSimilarityFusionWs([WsDict['Chromas'], WsDict['MFCCs']], K=pK, niters=niters, \
        reg_diag=reg_diag, reg_neighbs=reg_neighbs, \
        do_animation=do_animation, PlotNames=FeatureNames, \
        PlotExtents=[times[0], times[-1]]) 
    WsDict['Fused MFCC/Chroma'] = WFused2
    if plot_result:
        plotFusionResults(WsDict, {}, {}, times, win_fac)
        plt.savefig("%s_Plot.png"%filename, bbox_inches='tight')
    featuresret = {}
    for f in features:
        featuresret[f['name']] = f
    return {'Ws':WsDict, 'times':times, 'K':pK, 'features':featuresret}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True, help="Path to audio file")
    parser.add_argument('--do_animation', type=int, default=0, help='Save plots of iterations to disk')
    parser.add_argument('--plot_result', type=int, default=1, help='Plot the result of fusion')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate to use')
    parser.add_argument('--hop_length', type=int, default=512, help="Hop Size in samples")
    parser.add_argument('--win_fac', type=int, default=10, help="Number of windows to average in a frame.  If negative, then do beat tracking, and subdivide by |win_fac| times within each beat")
    parser.add_argument('--wins_per_block', type=int, default=20, help="Number of frames to stack in sliding window for every feature")
    parser.add_argument('--K', type=int, default=5, help="Number of nearest neighbors in similarity network fusion.  If -1, then autotune to sqrt(N) for an NxN similarity matrix")
    parser.add_argument('--reg_diag', type=float, default=1.0, help="Regularization for self-similarity promotion")
    parser.add_argument('--reg_neighbs', type=float, default=0.5, help="Regularization for direct neighbor similarity promotion")
    parser.add_argument('--niters', type=int, default=3, help="Number of iterations in similarity network fusion")
    parser.add_argument('--neigs', type=int, default=8, help="Number of eigenvectors in the graph Laplacian")
    parser.add_argument('--matfilename', type=str, default="out.mat", help="Name of the .mat file to which to save the results")
    parser.add_argument('--jsonfilename', type=str, default="out.json", help="Name of the .json file to which to save results for viewing in the GUI")
    parser.add_argument('--diffusion_znormalize', type=int, default=1, help="Whether to perform Z-normalization with diffusion maps to spread things out more")


    opt = parser.parse_args()
    res = getFusedSimilarity(opt.filename, sr=opt.sr, \
        hop_length=opt.hop_length, win_fac=opt.win_fac, wins_per_block=opt.wins_per_block, \
        K=opt.K, reg_diag=opt.reg_diag, reg_neighbs=opt.reg_neighbs, niters=opt.niters, \
        do_animation=opt.do_animation, plot_result=opt.plot_result, do_crema=True)
    sio.savemat(opt.matfilename, res)
    saveResultsJSON(opt.filename, res['times'], res['Ws'], opt.neigs, opt.jsonfilename, opt.diffusion_znormalize)
