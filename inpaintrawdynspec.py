import os
import sys
import glob

import numpy as np

import astropy.units as u
from astropy.time import Time

from scipy import interpolate
from scipy.optimize import curve_fit, minimize, least_squares, leastsq
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.linalg import toeplitz
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import median_filter
#from skimage.restoration import inpaint

import scintillation.dynspectools.dynspectools as dtools
import argparse

def FlatCovariance(SS):
    """
    Finds the covariance matrix of a flattened 2D array from its power spectrum

    Parameters
    ----------
    SS : (n, m) array
         Power spectrum of the underlying signal
    
    Returns
    -------
    Smat : (n * m, n * m) array
           2D covariance matrix
    """
    nx = SS.shape[1]
    ny = SS.shape[0]
    ACOR = np.fft.ifft2(SS).real
    ##Define correlation matrix for flattened array
    Smat = np.zeros((ny, nx, ny, nx), dtype=np.float)
    Smat = ACOR[np.mod(
        np.arange(ny)[:, np.newaxis, np.newaxis, np.newaxis] -
        np.arange(ny)[np.newaxis, np.newaxis, :, np.newaxis], ny),
                np.mod(
                    np.arange(nx)[np.newaxis, :, np.newaxis, np.newaxis] -
                    np.arange(nx)[np.newaxis, np.newaxis, np.newaxis, :], nx)]
    Smat = np.reshape(Smat, (nx * ny, nx * ny))
    return (Smat)

def SecspecWelch(ds, ntchunk, tchunk):
    window = np.hanning(tchunk)
    for i in range(ntchunk*2 - 1):
        trange = slice( i*tchunk//2, (i+2)*tchunk//2 )
        dsi = ds[trange] * window[:, np.newaxis]
        Si = np.abs(np.fft.fft2(dsi))**2.0
        if i == 0:
            S = np.copy(Si)
        else:
            S = S + Si
    S = S / (i+1)
    #S[0,0] = 0
    return S
    
def WienerFilter(arr, N, H, Smat=None, SS=None):
    """
    Wiener Filter Code optimized with conjugate gradient

    Parameters
    ----------
    arr : (n, m) array
        2D numpy array of the data
    N : (n, m) array
        2D array of the noise variance at each point
    H : (n, m) array
        2D array of mask at each point. 0 if mask, 1 otherwise.
    Smat : (n * m, n x m) array
        Covariance matrix of the flattened signal
    SS : (n, m)
        Power spectrum of the underlying signal, used to calculate Smat if not provided
    
    Returns
    -------
    Smat : (n, m) array
           2D filtered data
    """
    nx = arr.shape[1]
    ny = arr.shape[0]
    if Smat is None:
        Smat = FlatCovariance(SS)
    N = sparse.diags(N.ravel(), format='dia')
    H = sparse.diags(H.astype(np.float).ravel(), format='dia')

    ## Filter operator
    x = sparse.linalg.cg(H @ Smat @ H.T + N, arr.ravel().T)[0]
    filt = np.reshape((Smat @ H.T @ x).T, (ny, nx))
    return (filt)

def MosaicDynspec(dynspec, mask, noise, ntchunk, nfchunk, intrinsic=False):
    """
    Bin dynamic spectra as much as possible while still resolving SS in tau, ft
    """
    
    tchunk = (dynspec.shape[0]//ntchunk)
    fchunk = (dynspec.shape[1]//nfchunk)
    
    dyn_filtered = np.copy(dynspec)
    
    if intrinsic:
        tprof = (dynspec*mask).mean(-1)
        mask[tprof<=0] = 0
        dyn_filtered[tprof<=0, :] = 0
        tprof[tprof<=0] = 0
        
        mask_orig = np.copy(mask)
        mask = mask*tprof[:, np.newaxis]
        
    for j in range(nfchunk):
        frange = slice( j*fchunk, (j+1)*fchunk )
        S = SecspecWelch(dyn_filtered[:,frange], ntchunk, tchunk)
        for i in range(ntchunk):
            trange = slice( i*tchunk, (i+1)*tchunk )
        
            print("F{0}/{1}, T{2}/{3}".format(j+1, nfchunk, i+1, ntchunk) )                
 
            ds = dyn_filtered[trange, frange]
            ms = mask[trange, frange]
            NN = np.copy(noise[trange, frange])*ds.size
            #S = np.abs(np.fft.fft2(ds))**2.0
            filt=WienerFilter(ds, NN, ms, SS=S).real
            dyn_filtered[trange, frange] = filt
            if intrinsic or len(bpass)> 1:
                mask[trange, frange] = mask_orig[trange, frange]
    return dyn_filtered



import argparse

def main(raw_args=None):

    parser = argparse.ArgumentParser(description='Inpaint Dynamic Spectra using Wiener Filter')
    parser.add_argument("-fname",  type=str)
    #parser.add_argument("-nt", default=1, type=int)
    parser.add_argument("-tsize", default=180, type=int)
    parser.add_argument("-nf", default=8, type=int)
    #parser.add_argument("-bpass", default=False, action='store_true')
    parser.add_argument("-intrinsic", default=False, action='store_true')
    parser.add_argument("-v", default=False, action='store_true')
    
    a = parser.parse_args(raw_args)
    
    infile = a.fname
    #ntchunk = a.nt
    nfchunk = a.nf
    tsize = a.tsize
    intrinsic = a.intrinsic
    verbose = a.v
    #bpass = a.bpass
    
    if verbose:
        print("Opening {0}".format(infile))
    dynspec, dserr, t, F, psrname = dtools.read_psrflux(infile)
    ntchunk = int(dynspec.shape[0]//tsize) + 1
    #ntchunk = np.max([1, ntchunk])

    prefix = infile.split("dynspec")[0]
    psrfluxname = prefix + 'filtered.dynspec'
    if verbose:
        print("Filtered Spectrum will be written to {0}".format(psrfluxname))
    
    mask = np.ones_like(dynspec)
    mask[abs(dynspec) < 1e-9] = 0
    
    t0 = t[0]
    T = Time(t0, format='mjd')
    
    # Block sizes beyond ~100x100 become too cumbersome
    if verbose:
        print("Dynspec has shape {0}".format(dynspec.shape) )
    
    tchunk = dynspec.shape[0]//ntchunk
    dynspec = dynspec[:tchunk*ntchunk]
    dserr = dserr[:tchunk*ntchunk]
    Nstd = np.nanstd(dserr)
    N = np.ones_like(dynspec) * Nstd**2.0
    mask = mask[:tchunk*ntchunk]
    taxis = t[:tchunk*ntchunk]
    
    dynmask = dynspec*mask
    if verbose:
        print("Using blocks of size {0} nt, {1} nf".format(ntchunk, nfchunk))
    
    M = np.copy(mask)
    D = np.copy(dynmask)
    #mask_bh = 1 - mask
    #inpainted = inpaint.inpaint_biharmonic(D, mask_bh)
    inpainted = median_filter(D, [10,40])
    D[mask<0.5] = inpainted[mask<0.5]

    mn = np.mean(D)
    D = D / mn
    
    filt_fill = MosaicDynspec(D, M, N, ntchunk, nfchunk, intrinsic=intrinsic)
    
    dtools.write_psrflux(filt_fill, M, F, taxis, psrfluxname, psrname=psrname)
    return psrfluxname

if __name__ == "__main__":
    main()

