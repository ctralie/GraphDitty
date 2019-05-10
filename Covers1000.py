"""
Programmer: Chris Tralie
Purpose: Code for managing covers 1000 dataset
"""
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from CSMSSMTools import *
from Covers import *

def getCovers1000AudioFilename(filePrefix):
    audiofile = glob.glob("%s*"%filePrefix)
    filename = filePrefix
    for f in audiofile:
        if not f[-3::] == "txt" and not f[-3::] == "mat":
            filename = f
            break
    return filename

#Get the list of songs
def getCovers1000SongPrefixes(verbose = False):
    AllSongs = []
    for i in range(1, 396):
        songs = glob.glob("Covers1000/%i/*.txt"%i)
        songs = sorted([s[0:-4] for s in songs])
        if verbose:
            print(songs)
            print(sorted(songs))
            print("\n\n")
        AllSongs += songs
    return AllSongs

def getCovers1000Ks():
    import glob
    Ks = []
    for i in range(1, 396):
        songs = glob.glob("Covers1000/%i/*.txt"%i)
        Ks.append(len(songs))
    return Ks

if __name__ == '__main__':
    filenames = [getCovers1000AudioFilename(prefix) for prefix in getCovers1000SongPrefixes()]
    dim = 512
    norm_per_path = False
    #similarity_images, scattering_coeffs = get_scattering_corpus(filenames, dim=dim, norm_per_path = norm_per_path, do_plot=True)
    similarity_images, scattering_coeffs = get_haar_corpus(filenames, dim=dim, neigs=10, do_plot=False)
    DL2 = getSSM(similarity_images)
    similarity_images = None
    DScattering = getSSM(scattering_coeffs)
    scattering_coeffs = None
    sio.savemat("covers1000.mat", {"DL2":DL2, "DScattering":DScattering, "dim":dim})

    normstr = {True:"_Norm", False:""}
    Ks = getCovers1000Ks()
    fout = open("Covers1000ResultsLargeScale.html", "a")
    S = np.minimum(DL2, DL2.T)
    getEvalStatistics(-S, Ks, [1, 25, 50, 100], fout, "L2_%i"%(dim))
    S = np.minimum(DScattering, DScattering.T)
    getEvalStatistics(-S, Ks, [1, 25, 50, 100], fout, "Scattering_%i%s"%(dim, normstr[norm_per_path]))
    fout.close()