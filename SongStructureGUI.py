#Programmer: Chris Tralie
#Purpose: To extract similarity alignments for use in the GUI
import numpy as np
import os
import scipy.misc
import matplotlib.pyplot as plt
import json
import base64

def getBase64File(filename):
    fin = open(filename, "rb")
    b = fin.read()
    b = base64.b64encode(b)
    fin.close()
    return b.decode("ASCII")

def getBase64PNGImage(pD, cmapstr, logfloor = 0):
    """
    Get an image as a base64 string
    """
    D = np.array(pD)
    if logfloor > 0:
        D = np.log(D + logfloor)
    c = plt.get_cmap(cmapstr)
    D = D-np.min(D)
    D = np.round(255.0*D/np.max(D))
    C = c(np.array(D, dtype=np.int32))
    scipy.misc.imsave("temp.png", C)
    b = getBase64File("temp.png")
    os.remove("temp.png")
    return "data:image/png;base64, " + b

#http://stackoverflow.com/questions/1447287/format-floats-with-standard-json-module
class PrettyFloat(float):
    def __repr__(self):
        return '%.4g' % self
def pretty_floats(obj):
    if isinstance(obj, float):
        return PrettyFloat(obj)
    elif isinstance(obj, dict):
        return dict((k, pretty_floats(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return map(pretty_floats, obj)
    return obj

def saveResultsJSON(filename, times, W, v, jsonfilename):
    Results = {'songname':filename, 'times':times.tolist()}
    print("Saving results...")
    #Add music as base64 files
    path, ext = os.path.splitext(filename)
    Results['audio'] = "data:audio/%s;base64, "%ext[1::] + getBase64File(filename)
    WOut = np.array(W)
    np.fill_diagonal(WOut, 0)
    Results['W'] = getBase64PNGImage(WOut, 'afmhot', 5e-2)
    Results['dim'] = W.shape[0]
    
    # Resize the eigenvectors so they're easier to see
    fac = 10
    vout = np.zeros((v.shape[1]*10, v.shape[0]))
    for i in range(fac):
        vout[i::fac, :] = v.T
    Results['v'] = getBase64PNGImage(vout, 'afmhot')
    Results['v_height'] = vout.shape[0]
    fout = open(jsonfilename, "w")
    fout.write(json.dumps(Results))
    fout.close()

if __name__ == '__main__':
    filename = "MJ.mp3"
    path, ext = os.path.splitext(filename)
    res = "data:audio/%s;base64, "%ext[1::] + getBase64File(filename)
    print(res)