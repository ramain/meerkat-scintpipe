import numpy as np
import argparse

import scintparsfromdynspec

def main(raw_args=None):

    parser = argparse.ArgumentParser(description='Inpaint Dynamic Spectra using Wiener Filter')
    parser.add_argument("-archivefile", default='', type=str)
    parser.add_argument("-dynspecfile", default='', type=str)
    parser.add_argument("-outdir", type=str)

    a = parser.parse_args(raw_args)

    archivefile = a.archivefile
    dynspecfile = a.dynspecfile
    outdir = a.outdir

    if archivefile:
        import dynspecfromarchive
        import inpaintrawdynspec
        rawdynspecfile = dynspecfromarchive.main(['-fname', '{0}'.format(archivefile), '-outdir', '{0}'.format(outdir)])
        dynspecfile = inpaintrawdynspec.main(['-fname', '{0}'.format(rawdynspecfile), '-tsize', '150', '-nf', '8', '-intrinsic'])
        
    plotprefix = scintparsfromdynspec.main(['-fname', '{0}'.format(dynspecfile), '-outdir', '{0}'.format(outdir)])
    
    import os
    os.system("python createpano.py {0}".format(plotprefix))

if __name__ == "__main__":
    main()
