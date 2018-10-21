import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
import numpy as np
import librosa
import mir_eval
import jams
import glob
import os
from CSMSSMTools import getCSM, getCSMCosine
from Laplacian import *
from SimilarityFusion import getW, doSimilarityFusionWs
from multiprocessing import Pool as PPool

"""
SALAMI NOTES
*1359 tracks total, 884 have two distinct annotators
"""

#https://jams.readthedocs.io/en/stable/jams_structure.html
#https://craffel.github.io/mir_eval/#mir_eval.hierarchy.lmeasure

JAMS_DIR = 'salami-data-public-jams-multi'
AUDIO_DIR = 'salami-data-public-audio'

def jam_annos_to_lists(coarse, fine):
    """
    Given a list of annotations from a jam file, convert it into
    a format to pass to mir_eval
    Parameters
    ----------
    coarse: Annotation object
        An annotation object at the coarse level
    fine: Annotation object
        An annotation object at the fine level
    Returns
    -------
    intervals_hier: 2-element list of ndarray (N, 2)
        List of intervals (in seconds) of annotations
    labels_hier: 2-element list of N-length lists of strings
        Corresponding segment labels for annotations
    """
    intervals_hier = [[], []]
    labels_hier = [[], []]
    for i, anno in enumerate([coarse, fine]):
        for val in anno.data:
            intervals_hier[i].append(np.array([val.time, val.time+val.duration]))
            labels_hier[i].append(val.value)
        intervals_hier[i] = np.array(intervals_hier[i])
    return intervals_hier, labels_hier

def get_inter_anno_agreement_par(filename):
    print(filename)
    jam = jams.load(filename)
    if len(jam.annotations) < 8:
        return np.array([-1, -1, -1])
    intervals_hier1, labels_hier1 = jam_annos_to_lists(jam.annotations[1], jam.annotations[2])
    intervals_hier2, labels_hier2 = jam_annos_to_lists(jam.annotations[4], jam.annotations[5])
    l_precision, l_recall, l_measure = mir_eval.hierarchy.lmeasure(intervals_hier1, labels_hier1, intervals_hier2, labels_hier2)
    return np.array([l_precision, l_recall, l_measure])


def get_inter_anno_agreement(NThreads = 12):
    """
    Trying to replicate figure 11 in the Frontiers paper
    """
    if not os.path.exists("interanno.mat"):
        parpool = PPool(NThreads)
        filenames =  glob.glob("%s/*.jams"%JAMS_DIR)
        res = parpool.map(get_inter_anno_agreement_par, (filenames))
        sio.savemat("interanno.mat", {"res":res})
    res = sio.loadmat("interanno.mat")["res"]
    res = res[res[:, 0] > -1, :]
    sns.kdeplot(res[:, -1], shade=True)
    plt.xlim([0, 1])
    plt.title("L-Measure Inter-Annotator Agreement")
    plt.xlabel("L-Measure")
    plt.ylabel("Probability Density")
    plt.show()

def compute_features(num, do_plot=False):
    matfilename = "%s/%i/results.mat"%(AUDIO_DIR, num)
    if os.path.exists(matfilename):
        print("Skipping %i"%num)
        return
    filename = "%s/%i/audio.mp3"%(AUDIO_DIR, num)
    print("Doing %i..."%num)
    
    # Step 1: Initialize Feature/Fusion Parameters
    sr=22050
    hop_length=512
    win_fac=10
    wins_per_block=20
    K=10
    reg_diag=1.0
    reg_neighbs=0.5
    niters=10
    neigs=10
    
    y, sr = librosa.load(filename, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128, hop_length=hop_length)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=20)
    
    # Step 2: Compute features in intervals evenly spaced by the hop size
    # but average within "win_fac" intervals of hop_length
    nHops = int((y.size-hop_length*win_fac*wins_per_block)/hop_length)
    intervals = np.arange(0, nHops, win_fac)
    intervals = librosa.util.fix_frames(intervals, x_min=0, x_max=min(mfcc.shape[1], chroma.shape[1]))
    chroma = librosa.util.sync(chroma, intervals)
    mfcc = librosa.util.sync(mfcc, intervals)

    n_frames = min(chroma.shape[1], mfcc.shape[1])
    chroma = chroma[:, :n_frames]
    mfcc = mfcc[:, :n_frames]

    # Step 3: Do a delay embedding and compute SSMs
    XChroma = librosa.feature.stack_memory(chroma, n_steps=wins_per_block, mode='edge').T
    XMFCC = librosa.feature.stack_memory(mfcc, n_steps=wins_per_block, mode='edge').T
    DChroma = getCSMCosine(XChroma, XChroma) #Cosine distance
    DMFCC = getCSM(XMFCC, XMFCC) #Euclidean distance

    # Step 4: Run similarity network fusion
    FeatureNames = ['MFCCs', 'Chromas', 'Fused']
    Ds = [DMFCC, DChroma]
    Ws = [getW(D, K) for D in Ds]
    time_interval = hop_length*win_fac/float(sr)
    WFused = doSimilarityFusionWs(Ws, K=K, niters=niters, \
        reg_diag=reg_diag, reg_neighbs=reg_neighbs) 
    Ws.append(WFused)

    # Step 5: Compute Laplacian eigenvectors and perform spectral clustering
    # at different resolutions
    vs = []
    
    for W in Ws:
        _, v, _ = getSymmetricLaplacianEigsDense(W, neigs)
        vs.append(v)

    if do_plot:
        PlotExtents = [0, time_interval*WFused.shape[0]]
        fig = plt.figure(figsize=(8*len(Ws), 6))
        for i, (W, v, name) in enumerate(zip(Ws, vs, FeatureNames)):
            WShow = np.array(W)
            np.fill_diagonal(WShow, 0)
            plt.subplot2grid((1, 8*len(Ws)), (0, i*8), colspan=7)
            plt.imshow(np.log(5e-2+WShow), interpolation = 'nearest', cmap = 'afmhot', \
            extent = (PlotExtents[0], PlotExtents[1], PlotExtents[1], PlotExtents[0]))
            plt.title("%s Similarity Matrix"%name)
            plt.xlabel("Time (sec)")
            plt.ylabel("Time (sec)")
            plt.subplot2grid((1, 8*len(Ws)), (0, i*8+7))
            plt.imshow(v, cmap='afmhot', interpolation = 'nearest', aspect='auto', \
                extent=(0, v.shape[1], PlotExtents[1], PlotExtents[0]))
            plt.title("Laplacian")
            plt.xlabel("Eigenvector Num")
            plt.xticks(0.5 + np.arange(v.shape[1]), ["%i"%(i+1) for i in range(v.shape[1])])
        plt.tight_layout()
        figpath = "%s/%i/Fusion.png"%(AUDIO_DIR, num)
        print("Saving to %s"%figpath)
        plt.savefig(figpath, bbox_inches='tight')
        plt.close(fig)


def run_audio_experiments(NThreads = 12):
    songnums = [int(s) for s in os.listdir(AUDIO_DIR)]
    print(songnums)
    pass

if __name__ == '__main__':
    #get_inter_anno_agreement()
    #run_audio_experiments()
    compute_features(2, True)