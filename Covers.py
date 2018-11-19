import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from SongStructure import *
from CSMSSMTools import *
from scattering.scattering2d import Scattering2D

## Global fusion variables
sr=22050
hop_length=512
win_fac=10
wins_per_block=20
K=10
reg_diag=1.0
reg_neighbs=0.0
niters=10
neigs=10
do_mfcc = True
do_chroma = True
do_tempogram = True
do_crema = True
precomputed_crema = True


def get_scattering_corpus(filenames, dim = 512, norm_per_path = True, do_plot = False):
    """
    Get the scattering transform on resized SSMs from a corpus of music
    Parameters
    ----------
    filenames: list of N strings
        Paths to all files in the corpus
    dim: int
        Dimension to which to uniformly rescale SSMs (power of 2)
    norm_per_path: boolean
        Whether to normalize each path
    do_plot: boolean
        Whether to plot the fused SSMs and save them to disk
    Returns
    -------
    (similarity_images: ndarray(N, dim*dim)
        An array of self-similarity matrices for all 
    scattering_coeffs: ndarray(N, 81*dim*dim/16)
        An array of scattering coefficients for each song
    """

    # Initialize the scattering transform
    N = len(filenames)
    print("Initializing scattering transform...")
    tic = time.time()
    J = 4
    L = 8
    NPaths = L*L*J*(J-1)/2 + J*L + 1
    scattering = Scattering2D(M=dim, N=dim, J=J, L=L).cuda()
    print("Elapsed Time: %.3g"%(time.time()-tic))
    similarity_images = np.zeros((N, dim*dim)) # All similarity images
    ITemp = torch.zeros((1, 1, dim, dim))
    scattering_coeffs = np.zeros((N, int(NPaths*dim*dim/(2**(2*J)))), dtype=np.float32)
    for i, filename in enumerate(filenames):
        ## Step 1: Compute fused similarity matrix for the song
        print("Computing similarity fusion for song %i of %i..."%(i+1, N))
        res = getFusedSimilarity(filename, sr, hop_length, win_fac, wins_per_block, K, \
                        reg_diag, reg_neighbs, niters, False, False, \
                        do_mfcc = do_mfcc, do_chroma = do_chroma, do_tempogram = do_tempogram, \
                        do_crema=do_crema, precomputed_crema = precomputed_crema)
        W = res['Ws']['Fused']
        # Resize to a common dimension
        W = imresize(W, (dim, dim))
        similarity_images[i, :] = np.array(W.flatten(), dtype=np.float32)

        ## Step 2: Perform the 2D scattering transform
        ITemp[0, 0, :, :] = torch.from_numpy(W)
        resi = scattering(ITemp.cuda()).to("cpu").numpy()
        if norm_per_path:
            # Normalize coefficients in a path
            for ipath in range(resi.shape[2]):
                path = resi[0, 0, ipath, :, :]
                norm = np.sqrt(np.sum(path**2))
                if norm > 0:
                    resi[0, 0, ipath, :, :] /= norm
        scattering_coeffs[i, :] = np.array(resi.flatten(), dtype=np.float32)
        if do_plot:
            fig = plotFusionResults(res['Ws'], {}, {}, res['times'], win_fac)
            plt.savefig("%s_Similarity.png"%filename, bbox_inches='tight')
            plt.close(fig)
    return (similarity_images, scattering_coeffs)


def getEvalStatistics(ScoresParam, Ks, topsidx, fout = None, name = None):
    """
    Return evaluation statistics on a 
    Parameters
    ----------
    ScoresParam: ndarray(N, N)
        A N x N similarity matrix between all N songs in a corpus
    Ks: list (int)
        Number of songs in each clique, in order of presentation of cliques
        Numbers in this list sum to N
    topsidx: list (int)
        Compute number of songs in top x, for x in topsidx
    fout: File handle
        Handle to HTML file to which to output results
    name: string
        Name of the results
    """
    Scores = np.array(ScoresParam)
    N = Scores.shape[0]
    #Compute MR, MRR, MAP, and Median Rank
    #Fill diagonal with -infinity to exclude song from comparison with self
    np.fill_diagonal(Scores, -np.inf)
    idx = np.argsort(-Scores, 1) #Sort row by row in descending order of score
    ranks = np.zeros(N)
    startidx = 0
    kidx = 0
    for i in range(N):
        if i >= startidx + Ks[kidx]:
            startidx += Ks[kidx]
            kidx += 1
        print(startidx)
        for k in range(N):
            diff = idx[i, k] - startidx
            if diff >= 0 and diff < Ks[kidx]:
                ranks[i] = k+1
                break
    print(ranks)
    MR = np.mean(ranks)
    MRR = 1.0/N*(np.sum(1.0/ranks))
    MDR = np.median(ranks)
    print("MR = %g\nMRR = %g\nMDR = %g\n"%(MR, MRR, MDR))
    if fout:
        fout.write("<tr><td>%s</td><td>%g</td><td>%g</td><td>%g</td>"%(name, MR, MRR, MDR))
        tops = np.zeros(len(topsidx))
        for i in range(len(tops)):
            tops[i] = np.sum(ranks <= topsidx[i])
            print("Top-%i: %i"%(topsidx[i], tops[i]))
            fout.write("<td>%i</td>"%tops[i])
        fout.write("</tr>\n\n")
    return (MR, MRR, MDR, tops)
