import crema
import librosa
import sys
import warnings
import numpy as np
import scipy.io as sio
import subprocess
from multiprocessing import Pool as PPool
from SongStructure import *
from SalamiExperiments import *
model = crema.models.chord.ChordModel()

def compute_crema(filename):
    print("Doing crema on %s"%filename)
    matfilename = "%s_crema.mat"%filename
    subprocess.call([FFMPEG_BINARY, "-i", filename, "-ar", "44100", "-ac", "1", "%s.wav"%filename])
    _, y44100 = sio.wavfile.read("%s.wav"%filename)
    y44100 = y44100/2.0**15
    os.remove("%s.wav"%filename)

    data = model.outputs(y=y44100, sr=44100)
    sio.savemat(matfilename, data)

def compute_all_crema_salami(NThreads = 12):
    """
    Precompute all crema features in the SALAMI dataset
    """
    # Disable inconsistent hierarchy warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    filenames = ["%s/%s/audio.mp3"%(AUDIO_DIR, s) for s in os.listdir(AUDIO_DIR)]
    if NThreads > -1:
        parpool = PPool(NThreads)
        parpool.map(compute_crema, (filenames))
    else:
        for filename in filenames:
            compute_crema(filename)

def compute_all_crema_covers1000(NThreads = 12):
    """
    Precompute all crema features in the SALAMI dataset
    """
    from Covers1000 import getCovers1000AudioFilename, getCovers1000SongPrefixes
    # Disable inconsistent hierarchy warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    filenames = [getCovers100AudioFilename(p) for p in getCovers1000SongPrefixes()]
    if NThreads > -1:
        parpool = PPool(NThreads)
        parpool.map(compute_crema, (filenames))
    else:
        for filename in filenames:
            compute_crema(filename)

if __name__ == '__main__':
    #compute_all_crema_salami(-1)
    #compute_all_crema_covers1000(-1)
    compute_crema("MJ.mp3")