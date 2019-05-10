import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
from kymatio import Scattering2D
import time
from SongStructure import *
from CSMSSMTools import *
from Laplacian import *
from skimage import filters
from Haar import *

## Global fusion variables
lapfn = getRandomWalkLaplacianEigsDense
specfn = lambda v, dim, times: spectralClusterSequential(v, dim, times, rownorm=False)
sr=22050
hop_length=512
win_fac=10
wins_per_block=20
K=3
reg_diag=1.0
reg_neighbs=0.0
niters=10
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
    J = 6
    L = 8
    NPaths = L*L*J*(J-1)/2 + J*L + 1
    scattering = Scattering2D(shape=(dim, dim), J=J, L=L).cuda()
    print("Elapsed Time: %.3g"%(time.time()-tic))
    similarity_images = np.zeros((N, dim*dim)) # All similarity images
    ITemp = torch.zeros((1, 1, dim, dim))
    scattering_coeffs = np.zeros((N, int(NPaths*dim*dim/(2**(2*J)))), dtype=np.float32)
    for i, filename in enumerate(filenames):
        ## Step 1: Compute fused similarity matrix for the song
        print("Computing similarity fusion for song %i of %i..."%(i+1, N))
        matfilename = "%s_SSM_K%i.mat"%(filename, K)
        if not os.path.exists(matfilename):
            # Cache SSM computation
            res = getFusedSimilarity(filename, sr, hop_length, win_fac, wins_per_block, K, \
                            reg_diag, reg_neighbs, niters, False, False, \
                            do_mfcc = do_mfcc, do_chroma = do_chroma, do_tempogram = do_tempogram, \
                            do_crema=do_crema, precomputed_crema = precomputed_crema)
            W = res['Ws']['Fused']
            # Resize to a common dimension
            W = imresize(W, (dim, dim))
            sio.savemat(matfilename, {"W":W})
            if do_plot:
                fig = plotFusionResults(res['Ws'], {}, {}, res['times'], win_fac)
                plt.savefig("%s_Similarity.png"%filename, bbox_inches='tight')
                plt.close(fig)
        W = sio.loadmat(matfilename)["W"]
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
    return (similarity_images, scattering_coeffs)


def get_haar_corpus(filenames, dim = 512, neigs=None, do_plot = False):
    """
    Get the Haar scattering transform on resized SSMs from a corpus of music
    Parameters
    ----------
    filenames: list of N strings
        Paths to all files in the corpus
    dim: int
        Dimension to which to uniformly rescale SSMs initially (power of 2)
    neigs: int
        Maximum number of eigenvectors to use in a binary clustering
        representation.  If None, then use real numbers
    do_plot: boolean
        Whether to plot the fused SSMs and save them to disk
    Returns
    -------
    """
    N = len(filenames)
    similarity_images = np.zeros((N, dim*dim)) # All similarity images
    scattering_coeffs = np.zeros_like(similarity_images)
    res = 4
    plt.figure(figsize=(res*3, res*2))
    for i, filename in enumerate(filenames):
        ## Step 1: Compute fused similarity matrix for the song
        print("Computing similarity fusion for song %i of %i..."%(i+1, N))
        matfilename = "%s_SSM_K%i.mat"%(filename, K)
        if not os.path.exists(matfilename):
            # Cache SSM computation
            res = getFusedSimilarity(filename, sr, hop_length, win_fac, wins_per_block, K, \
                            reg_diag, reg_neighbs, niters, False, False, \
                            do_mfcc = do_mfcc, do_chroma = do_chroma, do_tempogram = do_tempogram, \
                            do_crema=do_crema, precomputed_crema = precomputed_crema)
            W = res['Ws']['Fused']
            # Resize to a common dimension
            W = imresize(W, (dim, dim))
            sio.savemat(matfilename, {"W":W})
        W = sio.loadmat(matfilename)["W"]
        H = get_haarscattering(W)
        

        ## Step 2: Compute Laplacian eigenvectors and successive binary approximations
        if neigs:
            vs = lapfn(W)[:, 0:neigs]
            WLowRank = vs.dot(vs.T)
            L = specfn(vs, neigs, np.arange(W.shape[0]))['labels']
            WB = np.array((L[None, :] - L[:, None]) == 0, dtype=float)
            HB = get_haarscattering(WB)
            similarity_images[i, :] = np.array(WB.flatten(), dtype=np.float32)
            scattering_coeffs[i, :] = np.array(HB.flatten(), dtype=np.float32)
        else:
            similarity_images[i, :] = np.array(W.flatten(), dtype=np.float32)
            scattering_coeffs[i, :] = np.array(H.flatten(), dtype=np.float32)

        if do_plot:
            plt.clf()
            plt.subplot(232)
            plt.imshow(W)
            plt.title("Similarity Image")
            plt.subplot(233)
            plt.imshow(H)
            plt.title("Haar")
            if neigs:
                plt.subplot(234)
                plt.imshow(WLowRank)
                plt.title("Low Rank Approx")
                plt.subplot(235)
                plt.imshow(WB)
                plt.title("Binary Image(%.3g)"%((np.sum(WB>0))/float(WB.shape[0]**2)))
                plt.subplot(236)
                plt.imshow(HB)
                plt.title("Haar Binary(%.3g)"%((np.sum(HB>0))/float(HB.shape[0]**2)))
            plt.savefig("%s_Haar.png"%filename, bbox_inches='tight')
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
