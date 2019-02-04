import os

if __name__ == '__main__':
    folders = ['.']
    for folder in folders:
        files = os.listdir(folder)
        thisFiles = {}
        for f in files:
            f = "%s/%s"%(folder, f)
            parts = os.path.splitext(f)
            if parts[-1] == '.svg':
                thisFiles[os.path.getmtime(f)] = f
    
        for key in sorted(thisFiles):
            f = thisFiles[key]
            parts = os.path.splitext(f)
            fnew = "%s.pdf"%parts[0]
            print "Making ", fnew
            os.popen3("inkscape -D -z --file=%s --export-pdf=%s"%(f, fnew))
    

