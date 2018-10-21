import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
import numpy as np
import librosa
import mir_eval
import jams
import glob
import os
from multiprocessing import Pool as PPool

"""
SALAMI NOTES
*1359 tracks total, 884 have two distinct annotators
"""

#https://jams.readthedocs.io/en/stable/jams_structure.html
#https://craffel.github.io/mir_eval/#mir_eval.hierarchy.lmeasure

JAMS_DIR = 'salami-data-public-jams-multi'
AUDIO_DIR = 'salami-data-public-audio'
#mir_eval.hierarchy.lmeasure

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

if __name__ == '__main__':
    get_inter_anno_agreement()