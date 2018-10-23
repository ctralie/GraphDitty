"""
SALAMI NOTES
*1359 tracks total, 884 have two distinct annotators
"""
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
from scipy import stats
import numpy as np
import librosa
import mir_eval
import jams
import glob
import os
import sys
import warnings
import time
from CSMSSMTools import getCSM, getCSMCosine
from Laplacian import *
from SimilarityFusion import getW, doSimilarityFusionWs
from SongStructure import *
from multiprocessing import Pool as PPool

## Paths to dataset
JAMS_DIR = 'salami-data-public-jams-multi'
AUDIO_DIR = 'salami-data-public-audio'

## Global fusion variables
lapfn = getUnweightedLaplacianEigsDense
specfn = lambda v, dim, times: spectralClusterSequential(v, dim, times, rownorm=False)
sr=22050
hop_length=512
win_fac=-2
wins_per_block=40
K=10
reg_diag=1.0
reg_neighbs=0.5
niters=10
neigs=10



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

def compute_features(num, recompute=False):
    matfilename = "%s/%i/results.mat"%(AUDIO_DIR, num)
    jamsfilename = "%s/%i.jams"%(JAMS_DIR, num)
    if (not recompute) and (os.path.exists(matfilename) or (not os.path.exists(jamsfilename))):
        print("Skipping %i"%num)
        return
    filename = "%s/%i/audio.mp3"%(AUDIO_DIR, num)
    print("Doing %i..."%num)

    # Step 1: Compute feature-based similarity matrix and the matrix
    # fusing all of them
    res = getFusedSimilarity(filename, sr, hop_length, win_fac, wins_per_block, K, reg_diag, reg_neighbs, niters, False, False)
    Ws, times = res['Ws'], res['times']

    # Step 2: Compute Laplacian eigenvectors and perform spectral clustering
    # at different resolutions
    vs = {name:lapfn(Ws[name])[:, 1:neigs+1] for name in Ws}
    alllabels = {name:[specfn(vs[name], k, times) for k in range(2, neigs+1)] for name in Ws}
    #print("Elapsed time spectral clustering: %.3g"%(time.time()-tic))
    specintervals_hier = {name:[res['intervals_hier'] for res in alllabels[name]] for name in alllabels}
    speclabels_hier = {name:[res['labels_hier'] for res in alllabels[name]] for name in alllabels}

    ## Step 3: Compare to annotators and save results
    jam = jams.load(jamsfilename)    
    ret = {name:[] for name in Ws}
    for annidx in range(int(len(jam.annotations)/4)):
        for name in alllabels:
            intervals_hier, labels_hier = jam_annos_to_lists(jam.annotations[1+annidx*3], jam.annotations[2+annidx*3])
            # Make sure the labels end at the same place (extend the spectral clustering label if necessary)
            end = intervals_hier[-1][-1, 1]
            for i in range(len(specintervals_hier[name])):
                specintervals_hier[name][i][-1, 1] = end 
            l_precision, l_recall, l_measure = mir_eval.hierarchy.lmeasure(intervals_hier, labels_hier, specintervals_hier[name], speclabels_hier[name])
            ret[name] += [l_precision, l_recall, l_measure]
            print("Annotator %i song %i %s: p = %.3g, r = %.3g, l = %.3g"%(annidx, num, name, l_precision, l_recall, l_measure))
    sio.savemat(matfilename, ret)

    ## Step 4: Plot SSM, eigenvectors, and clustering at the finest level
    fig = plotFusionResults(Ws, vs, alllabels, times)
    figpath = "%s/%i/Fusion.png"%(AUDIO_DIR, num)
    print("Saving to %s"%figpath)
    plt.savefig(figpath, bbox_inches='tight')
    plt.close(fig)


def run_audio_experiments(NThreads = 12):
    """
    Run all of the SALAMI feature computation and fusion in parallel
    and save the results
    """
    # Disable inconsistent hierarchy warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    songnums = [int(s) for s in os.listdir(AUDIO_DIR)]
    songnums.remove(878)
    if NThreads > -1:
        parpool = PPool(NThreads)
        parpool.map(compute_features, (songnums))
    else:
        for num in songnums:
            compute_features(num)

def aggregate_experiments_results():
    """
    Load all of the results from the SALAMI experiment and plot
    the annotator agreements
    """
    # Step 1: Extract inter-annotator agreements
    interanno = sio.loadmat("interanno.mat")["res"]
    interanno = interanno[interanno[:, 0] > -1, :]

    # Step 2: Extract feature-based agreements
    names = ['MFCCs', 'Chromas', 'Fused']
    prls = {name:np.zeros((0, 3)) for name in names} # Dictionary of precison, recall, and l-scores
    idxs = [] #Indices of 

    for num in [int(s) for s in os.listdir(AUDIO_DIR)]:
        matfilename = '%s/%i/results.mat'%(AUDIO_DIR, num)
        if os.path.exists(matfilename):
            res = sio.loadmat(matfilename)
            nanno = 0
            for name in names:
                nres = res[name]
                nres = np.reshape(nres, (int(nres.size/3), 3))
                prls[name] = np.concatenate((prls[name], nres), 0)
                nanno = nres.shape[0]
            idxs += [num]*nanno
    idxs = np.array(idxs)
    res = {a:prls[a] for a in prls}
    res['idxs'] = idxs
    sio.savemat("allresults.mat", res)

    # Step 3: Plot distribution and KS-score of feature-based agreements
    # versus inter-annotator agreements
    plt.figure(figsize=(15, 5))
    for i, plotname in enumerate(['Precision', 'Recall', 'L-Measure']):
        plt.subplot(1, 3, i+1)
        legend = ['interanno']
        sns.kdeplot(interanno[:, i], shade=True)
        for name in names:
            prl = prls[name]
            sns.kdeplot(prl[:, i], shade=True)
            k = stats.ks_2samp(interanno[:, i], prl[:, i])[0]
            legend.append('%s, K=%.3g'%(name, k))
        plt.legend(legend)
        plt.title("SPAM %s"%plotname)
        plt.xlabel(plotname)
        plt.ylabel("Probability Density")
        plt.xlim([0, 1])
    plt.savefig("Results.svg", bbox_inches='tight')

    # Step 4: Plot distribution of improvements with fusion
    names = ['MFCCs', 'Chromas']
    plt.figure(figsize=(15, 5))
    for i, plotname in enumerate(['Precision', 'Recall', 'L-Measure']):
        plt.subplot(1, 3, i+1)
        for name in names:
            prl = prls[name]
            improvements = prls['Fused'][:, i]/prl[:, i]
            order = np.argsort(improvements)[0:10]
            s = ""
            for o in order:
                s += "\n%i: %.3g"%(idxs[o], improvements[o])
            print("Worst 10 %s %s: %s"%(name, plotname, s))
            print(improvements)
            #sns.kdeplot(improvements, shade=True)
        plt.legend(names)
        plt.title("SPAM %s Fusion Improvement"%plotname)
        plt.xlabel(plotname)
        plt.ylabel("Probability Density")
        #plt.gca().set_xscale("log")
    plt.savefig("Improvements.svg", bbox_inches='tight')


if __name__ == '__main__':
    #get_inter_anno_agreement()
    run_audio_experiments(NThreads=4)
    #aggregate_experiments_results()
    #compute_features(2, True)