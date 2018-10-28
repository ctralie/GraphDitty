"""
SALAMI NOTES
*1359 tracks total, 884 have two distinct annotators
"""
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
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
lapfn = getRandomWalkLaplacianEigsDense
specfn = lambda v, dim, times: spectralClusterSequential(v, dim, times, rownorm=False)
sr=22050
hop_length=512
win_fac=10
wins_per_block=20
K=-1
reg_diag=1.0
reg_neighbs=0.0
niters=10
neigs=10
REC_SMOOTH = 9



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
    Replicating the interanno part of figure 11 in the Frontiers paper
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

def compute_features(num, multianno_only = True, recompute=False):
    """
    Compute precision, recall, and l-measure for a set of features on a particular
    song in the SALAMI dataset, and save the results to a file called "results.mat"
    in the folder where the audio resides
    Parameters
    ----------
    num: int
        Song number in the SALAMI dataset
    multianno_only: boolean
        If true, only consider songs that have more than one annotation
    recompute: boolean
        Compute this again, even if "results.mat" already exists
    """
    matfilename = "%s/%i/results.mat"%(AUDIO_DIR, num)
    jamsfilename = "%s/%i.jams"%(JAMS_DIR, num)
    if (not recompute) and (os.path.exists(matfilename) or (not os.path.exists(jamsfilename))):
        print("Skipping %i because it has already been computed"%num)
        return
    filename = "%s/%i/audio.mp3"%(AUDIO_DIR, num)
    jam = jams.load(jamsfilename)    
    if multianno_only and len(jam.annotations) < 8:
        print("Skipping %i because there is only one annotation"%num)
        return
    print("Doing %i..."%num)

    # Step 1: Compute feature-based similarity matrix and the matrix
    # fusing all of them
    res = getFusedSimilarity(filename, sr, hop_length, win_fac, wins_per_block, K, reg_diag, reg_neighbs, niters, False, False)
    Ws, times = res['Ws'], res['times']
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    for name in Ws:
        Ws[name] = df(Ws[name], size=(1, REC_SMOOTH))
        np.fill_diagonal(Ws[name], 0)

    # Step 2: Compute Laplacian eigenvectors and perform spectral clustering
    # at different resolutions
    vs = {name:lapfn(Ws[name])[:, 1:neigs+1] for name in Ws}
    alllabels = {name:[specfn(vs[name], k, times) for k in range(2, neigs+1)] for name in Ws}
    #print("Elapsed time spectral clustering: %.3g"%(time.time()-tic))
    specintervals_hier = {name:[res['intervals_hier'] for res in alllabels[name]] for name in alllabels}
    speclabels_hier = {name:[res['labels_hier'] for res in alllabels[name]] for name in alllabels}

    ## Step 3: Compare to annotators and save results
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
    if len(jam.annotations) == 8:
        # Compute and store inter-annotator for convenience
        intervals_hier1, labels_hier1 = jam_annos_to_lists(jam.annotations[1], jam.annotations[2])
        intervals_hier2, labels_hier2 = jam_annos_to_lists(jam.annotations[4], jam.annotations[5])
        l_precision, l_recall, l_measure = mir_eval.hierarchy.lmeasure(intervals_hier1, labels_hier1, intervals_hier2, labels_hier2)
        ret['interanno'] = [l_precision, l_recall, l_measure]
    sio.savemat(matfilename, ret)

    ## Step 4: Plot SSM, eigenvectors, and clustering at the finest level
    intervals_hier1, labels_hier1 = jam_annos_to_lists(jam.annotations[1], jam.annotations[2])
    fig = plotFusionResults(Ws, vs, alllabels, times, win_fac, intervals_hier1, labels_hier1)
    if win_fac > 0:
        figpath = "%s/%i/Fusion.svg"%(AUDIO_DIR, num)
    else:
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
    if NThreads > -1:
        parpool = PPool(NThreads)
        parpool.map(compute_features, (songnums))
    else:
        for num in songnums:
            compute_features(num)

def aggregate_experiments_results(precomputed_name = "", multianno_only = True):
    """
    Load all of the results from the SALAMI experiment and plot
    the annotator agreements
    """
    # Step 1: Extract feature-based agreements
    names = ['MFCCs', 'Chromas', 'Tempogram', 'Crema', 'Fused Tgram/Crema', 'Fused MFCC/Chroma', 'Fused', 'interanno']
    prls = {name:np.zeros((0, 3)) for name in names} # Dictionary of precison, recall, and l-scores
    idxs = [] #Indices of 

    if len(precomputed_name) == 0:
        for num in [int(s) for s in os.listdir(AUDIO_DIR)]:
            matfilename = '%s/%i/results.mat'%(AUDIO_DIR, num)
            if os.path.exists(matfilename):
                res = sio.loadmat(matfilename)
                thisnanno = 0
                for name in names:
                    if name in res:
                        nres = res[name]
                        nres = np.reshape(nres, (int(nres.size/3), 3))
                        nanno = nres.shape[0]
                        thisnanno = max(thisnanno, nanno)
                        if (not (name == 'interanno')) and nanno < 2 and multianno_only:
                            continue
                        prls[name] = np.concatenate((prls[name], nres), 0)
                idxs += [num]*thisnanno
        idxs = np.array(idxs)
        print("idxs.shape = ", idxs.shape)
        res = {a:prls[a] for a in prls}
        res['idxs'] = idxs
        sio.savemat("allresults.mat", res)
    else:
        res = sio.loadmat(precomputed_name)
        idxs = res['idxs'].flatten()
        print("idxs.shape = ", idxs.shape)
        counts = {}
        for idx in idxs:
            if not idx in counts:
                counts[idx] = 0
            counts[idx] += 1
        to_keep = np.ones_like(idxs)
        for i, idx in enumerate(idxs):
            if counts[idx] < 2:
                to_keep[i] = 0
        print(to_keep.shape)
        res.pop('idxs')
        for name in names:
            res[name] = res[name][to_keep == 1, :]
        prls = res
    print("Plotting statistics for %i examples"%(res['MFCCs'].shape[0]/2))
    interanno = res['interanno']
    names.remove('interanno')
        

    # Step 2: Plot distribution and KS-score of feature-based agreements
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
            legend.append('%s, K=%.3g, Mean=%.3g'%(name, k, np.mean(prl[:, i])))
        plt.legend(legend)
        plt.title("Salami %s"%plotname)
        plt.xlabel(plotname)
        plt.ylabel("Probability Density")
        plt.xlim([0, 1])
        ymax = min(plt.gca().get_ylim()[1], 8)
        plt.ylim([0, ymax])
        plt.ylim([0, 5])
        
    plt.savefig("Results.svg", bbox_inches='tight')
    
    plt.clf()
    interanno = prls['interanno']
    ## Step 3: Scatter inter-annotator scores against fused scores
    for i, plotname in enumerate(['Precision', 'Recall', 'L-Measure']):
        plt.subplot(1, 3, i+1)
        prl = prls['Fused']
        plt.scatter(prl[0::2, i], interanno[:, i])
        plt.scatter(prl[1::2, i], interanno[:, i])
        plt.title("Salami %s"%plotname)
        plt.xlabel("Annotator-fused agreement")
        plt.ylabel("Annotator-annotator agreement")
    plt.savefig("Results_AnnotatorAgreement.png", bbox_inches='tight')

    ## Step 4: Report top 10 recall improvements of fusion over other features
    improvement = np.ones(prls['Fused'].shape[0])
    for name in names:
        if name == 'Fused' or name == 'interanno':
            continue
        improvement += prls[name][:, i]
    print("idxs.size = ", idxs.size)
    print("improvement.size = ", improvement.size)
    print(idxs[np.argsort(-improvement)][0:20])


if __name__ == '__main__':
    #get_inter_anno_agreement()
    #run_audio_experiments(NThreads=-1)
    aggregate_experiments_results()