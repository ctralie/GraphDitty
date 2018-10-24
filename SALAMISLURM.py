import numpy as np 
from SalamiExperiments import *
import argparse
import sys
import warnings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=0, help="Song number")
    opt = parser.parse_args()
    num = opt.num
    songnums = [int(s) for s in os.listdir(AUDIO_DIR)]
    songnums.remove(878)
    if num in songnums:
        if not sys.warnoptions:
            warnings.simplefilter("ignore")
        print("Doing %i"%num)
        compute_features(num)