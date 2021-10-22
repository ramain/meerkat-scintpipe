import os
import glob
import numpy as np
import subprocess 

d = np.loadtxt('census.files', dtype='str')
outfile = open("census_SNarchives.txt", 'w')

arcclass = np.loadtxt('pulsar_classification.txt', comments='#', dtype='str')
arcpsrs = arcclass[:,0]

for i in range(d.shape[0]):
    psr = d[i,0]
    if psr in arcpsrs:
        j = np.argwhere(arcpsrs==psr).squeeze()
        t0 = d[i,1]
        try:
            fname=glob.glob('/fred/oz005/users/aparthas/reprocessing_MK/TPA/{0}/{1}*/*/*/decimated/*zapTF.fluxcal'.format(psr, t0))
            fname = fname[0]
        except:
            fname=glob.glob('/fred/oz005/users/aparthas/reprocessing_MK/TPA/{0}/{1}*/*/*/decimated/*zapTF.ar'.format(psr, t0))
            fname =fname[0]
        
        a = subprocess.check_output( "psrstat -c dm,snr=pdmp,snr {0}".format(fname), shell=True).decode('UTF-8')
        sn = a.split()[-1][4:-2]
        dm = a.split()[-2][3:]
        outfile.write("{0} {1} {2} {3} {4}\n".format(psr, t0, dm, sn, arcclass[j,1]))

outfile.close()
