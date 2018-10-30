"""
Programmer: Chris Tralie
Purpose: To provide an interface for loading music, computing features, and
doing similarity fusion on those features to make a weighted adjacency matrix
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
import scipy.interpolate
import os
import librosa
import librosa.display
import argparse
import pandas
import mir_eval
from CSMSSMTools import getCSM, getCSMCosine
from SongStructure import *
from SalamiExperiments import *

def SalamiSSMFigure(num, cmap = 'magma_r'):
    ## Step 1: Show SSMs
    ret = compute_features(num, recompute=True)
    Ws, times = ret['Ws'], ret['times']
    nrows=2
    ncols=3
    resol = 2.6
    fig = plt.figure(figsize=(resol*ncols, resol*nrows))
    time_uniform = win_fac >= 0
    names = ['MFCCs', 'Chromas', 'Fused MFCC/Chroma', 'Tempogram', 'Crema', 'Fused']
    #names = ['MFCCs', 'Tempogram', 'Chromas', 'Crema', 'Fused MFCC/Chroma', 'Fused']
    plotnames = [n for n in names]
    plotnames[-2] = 'CREMA'
    plotnames[2] = 'MFCCs/Chromas Fused'
    plotnames[-1] = 'All Fused'

    for i, name in enumerate(names):
        W = Ws[name]
        floor = np.quantile(W.flatten(), 0.01)
        WShow = np.log(W+floor)
        pix = np.arange(W.shape[0])
        I, J = np.meshgrid(pix, pix)
        WShow[np.abs(I-J) < 3] = np.min(WShow)
        plt.subplot(nrows, ncols, i+1)
        row, col = np.unravel_index(i, (nrows, ncols))
        if time_uniform:
            plt.imshow(WShow, cmap =cmap, extent=(times[0], times[-1], times[-1], times[0]), interpolation='nearest')
        else:
            plt.pcolormesh(times, times, WShow, cmap = cmap)
            plt.gca().invert_yaxis()
        plt.title(plotnames[i])
        if row == nrows-1:
            plt.xlabel("Time (sec)")
        else:
            plt.gca().get_xaxis().set_visible(False)
        if col == 0:
            plt.ylabel("Time (sec)")
        else:
            plt.gca().get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig("Figures/%i_SSMs.svg"%num, bbox_inches='tight')


    ## Step 2: Show meet matrices for estimations and annotations
    interval = 0.25
    specintervals_hier = ret['specintervals_hier']
    speclabels_hier = ret['speclabels_hier']
    nrows=3
    fig = plt.figure(figsize=(resol*ncols, resol*nrows))
    for i, name in enumerate(names):
        row, col = np.unravel_index(i, (nrows, ncols))
        L = np.asarray(mir_eval.hierarchy._meet(specintervals_hier[name], speclabels_hier[name], interval).todense())
        times = interval*np.arange(L.shape[0])
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(L, cmap = 'gray_r', extent=(times[0], times[-1], times[-1], times[0]), interpolation='nearest')
        plt.title(plotnames[i])
        if row == nrows-2:
            plt.xlabel("Time (sec)")
        else:
            plt.gca().get_xaxis().set_visible(False)
        if col == 0:
            plt.ylabel("Time (sec)")
        else:
            plt.gca().get_yaxis().set_visible(False)
    jam = ret['jam']
    annotators = []
    annotators.append(jam_annos_to_lists(jam.annotations[4], jam.annotations[5]))
    annotators.append(jam_annos_to_lists(jam.annotations[1], jam.annotations[2]))
    for k in range(2):
        plt.subplot(3, 3, 7+k)
        intervals, labels = annotators[k]
        L = np.asarray(mir_eval.hierarchy._meet(intervals, labels, interval).todense())
        times = interval*np.arange(L.shape[0])
        plt.imshow(L, cmap = 'gray_r', extent=(times[0], times[-1], times[-1], times[0]), interpolation='nearest')
        plt.title("Annotator %i"%(k+1))
        plt.xlabel("Time (sec)")
        if col == 0:
            plt.ylabel("Time (sec)")
        else:
            plt.gca().get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig("Figures/%i_Meet.svg"%num, bbox_inches='tight')



def makeSNSPlot():
    plt.style.use('seaborn-darkgrid')
    res = sio.loadmat('allresults_K3.mat')
    scluster = pandas.read_csv('scluster_salami_all.csv')
    res['scluster'] = scluster[['MULTI_L-Precision', 'MULTI_L-Recall', 'MULTI_L-Measure']].values
    names = ['scluster', 'Fused', 'interanno']
    interanno = res['interanno']
    interanno2 = interanno[:, [1, 0, 2]]
    interanno = np.concatenate((interanno, interanno2), 0)

    # Plot distribution and KS-score of feature-based agreements
    # versus inter-annotator agreements
    fac = 0.55
    plt.figure(figsize=(fac*15, fac*5))
    ylims = [4.9, 3.2, 4.5]
    for i, plotname in enumerate(['L-Precision', 'L-Recall', 'L-Measure']):
        plt.subplot(1, 3, i+1)
        legend = []
        for name in names:
            prl = res[name]
            sns.kdeplot(prl[:, i], shade=True)
            if name == 'interanno':
                legend.append("interanno, mean = %.3g"%np.mean(interanno[:, i]))
            else:
                k = stats.ks_2samp(interanno[:, i], prl[:, i])[0]
                legend.append('%s, K=%.3g, Mean=%.3g'%(name, k, np.mean(prl[:, i])))
        plt.legend(legend)
        plt.title("Salami %s"%plotname)
        plt.xlabel(plotname)
        plt.ylabel("Probability Density")
        plt.xlim([0, 1])
        plt.ylim([0, ylims[i]])
    plt.savefig("Figures/SNSPlots.svg", bbox_inches='tight')

def makeTable():
    res3 = sio.loadmat('allresults_K3.mat')
    res10 = sio.loadmat('allresults_K10.mat')
    scluster = pandas.read_csv('scluster_salami_all.csv')
    scluster = scluster[['MULTI_L-Precision', 'MULTI_L-Recall', 'MULTI_L-Measure']].values
    interanno = res3['interanno']
    interanno2 = interanno[:, [1, 0, 2]]
    interanno = np.concatenate((interanno, interanno2), 0)

    # L-Precision Mean, L-Precision KS, L-Recall Mean, L-Recall KS, L-Score Mean, L-Score KS
    # interanno, scluster, k3 features, k10 features
    names = ['interanno', 'scluster']
    res = np.zeros((2, 6))
    res[0, 0::2] = np.mean(interanno, 0)
    for i in range(3):
        res[1, i*2] = np.mean(scluster[:, i])
        res[1, i*2+1] = stats.ks_2samp(interanno[:, i], scluster[:, i])[0]
    for name in ['MFCCs', 'Chromas', 'Tempogram', 'Crema', 'Fused MFCC/Chroma', 'Fused Tgram/Crema', 'Fused']:
        prl = res3[name]
        names.append(name)
        res = np.concatenate((res, np.zeros((1, 6))), 0)
        for i in range(3):
            res[-1, i*2] = np.mean(prl[:, i])
            res[-1, i*2+1] = stats.ks_2samp(interanno[:, i], prl[:, i])[0]
    res = np.concatenate((res, np.zeros((1, 6))), 0)
    prl = res10['Fused']
    for i in range(3):
        res[-1, i*2] = np.mean(prl[:, i])
        res[-1, i*2+1] = stats.ks_2samp(interanno[:, i], prl[:, i])[0]
    names.append('Fused $K=10$')
    
    df = pandas.DataFrame(res, index=names, columns = ['Mean Precision', 'KS Precision', 'Mean Recall', 'KS Recall', 'Mean L-Score', 'KS L-Score'])
    print(df.to_latex(float_format='%.3g'))

if __name__ == '__main__':
    #SalamiSSMFigure(num=936)
    #makeSNSPlot()
    makeTable()