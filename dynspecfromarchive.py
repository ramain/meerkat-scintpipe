import glob
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import ICRS, Galactic, FK4, FK5

from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import scintillation.dynspectools.slowft as slowft
import scintillation.dynspectools.dynspectools as dtools
import psrchive

import argparse

def plot_secspec(dynspec, freqs, t0, xlim=None, ylim=None, bintau=2, dt=4*u.s, vm=3., sft=True,
                binft=1, aspect=(10,10), plot_title=False, curv=0):

    """
    dynspec:  array with units [time, frequency]
    freqs: array of frequencies in MHz
    t0: Starting time in mjd.  Assumed 10 second bins
    xlim: xaxis limits in mHz
    ylim: yaxis limits in mus
    bintau:  Binning factor of SS in tau, for plotting purposes
    """
    
    t0 = Time(t0, format='mjd', precision=0)

    # Get frequency and time info for plot axes
    bw = freqs[-1] - freqs[0]
    df = (freqs[1]-freqs[0])*u.MHz
    nt = dynspec.shape[0]
    T = (nt * dt).to(u.min).value
    
    # Bin dynspec in time, frequency 
    # ONLY FOR PLOTTING
    dynspec = dynspec / np.std(dynspec)
    mn = np.mean(dynspec)
    
    bint=1
    nbin = dynspec.shape[0]//bint
    dspec_plot = dynspec[:nbin*bint].reshape(nbin, bint, dynspec.shape[-1]).mean(1)

    # 2D power spectrum is the Secondary spectrum
    if sft:
        print('Slow FT')
        CS = dtools.create_secspec(dynspec, freqs, fref=max(freqs), pad=1, taper=0)
        S = np.abs(CS)**2.0
    else:
        CS = np.fft.fft2(dynspec)
        S = np.fft.fftshift(CS)
        S = np.abs(S)**2.0
    Sb = S.reshape(-1,S.shape[1]//bintau, bintau).mean(-1)
    
    ntbin = Sb.shape[0]//binft
    Sb = Sb[:ntbin*binft].reshape(ntbin, binft, -1).mean(1)    
    Sb = np.log10(Sb)
    
    # Calculate the confugate frequencies (time delay, fringe rate), only used for plotting
    ft = np.fft.fftfreq(S.shape[0], dt)
    ft = np.fft.fftshift(ft.to(u.mHz).value)

    tau = np.fft.fftfreq(S.shape[1], df)
    tau = np.fft.fftshift(tau.to(u.microsecond).value)    
    
    ftb = ft[:ntbin*binft].reshape(-1, binft).mean(-1)
    Sb = Sb - np.mean(Sb[abs(ftb)>50], axis=0, keepdims=True)
    
    slow = np.median(Sb)-0.2
    shigh = slow + vm

    # Not the nicest, have a set of different plots it can produce
    plt.figure(figsize=aspect)
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    ax3 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)

    plt.subplots_adjust(wspace=0.1)

    # Plot dynamic spectrum image

    ax2.imshow(dspec_plot.T, aspect='auto', vmax=mn+5, vmin=mn-2, origin='lower',
                extent=[0,T,min(freqs), max(freqs)], cmap='viridis')
    ax2.set_xlabel('time (min)', fontsize=16)
    ax2.set_ylabel('freq (MHz)', fontsize=16)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    # Plot Secondary spectrum
    ax3.imshow(Sb.T, aspect='auto', vmin=slow, vmax=shigh, origin='lower',
               extent=[min(ft), max(ft), min(tau), max(tau)], interpolation='nearest',
              cmap='viridis')
    ax3.set_xlabel(r'$f_{D}$ (mHz)', fontsize=16)
    ax3.set_ylabel(r'$\tau$ ($\mu$s)', fontsize=16) 

    if curv:
        plt.plot(ft, curv*ft**2.0, linestyle='dotted', color='tab:blue')
    
    if xlim:
        ax3.set_xlim(-xlim, xlim)
    if ylim:
        ax3.set_ylim(-ylim, ylim)
    else:
        ax3.set_ylim(min(tau), max(tau))
    if plot_title:
        ax3.set_title(' {0}'.format(psrname), fontsize=18)
        ax2.set_title(' {0}'.format(t0.isot), fontsize=18)
    return CS, ft, tau

def main(raw_args=None):

    parser = argparse.ArgumentParser(description='temp description')
    parser.add_argument("-fname", type=str)
    parser.add_argument("-outdir", type=str)
    
    a = parser.parse_args(raw_args)
    fname = a.fname
    outdir = a.outdir
    
    I, F, t, psrname, tel = dtools.readpsrarch(fname)
    T = Time(t, format='mjd')
    dt = T[1].unix - T[0].unix
    
    foldspec, flag, mask, bg, bpass = dtools.clean_foldspec(I, plots=True, 
                                     apply_mask=False, rfimethod='var', 
                                     flagval=6, offpulse='True', tolerance=0.6)
    fs_clean = foldspec * mask[...,np.newaxis]
    template = fs_clean.mean(0).mean(0)
    dynspec = dtools.create_dynspec(fs_clean, template=template)
    dyn_clean = dynspec * mask
    
    if dyn_clean.shape[1] == 1024:
        dyn_clean = dyn_clean[:, 48:-48]
        F=F[48:-48]
    
    Funits = F*u.MHz
    Tunits = T[0] + np.arange(dyn_clean.shape[0])*dt*u.s
    
    note='raw dynspec created by psrchive and clean_foldspec in dynspectools by RMain'
    
    dtools.plot_diagnostics(fs_clean, flag, mask)
    
    psrfluxname = '{0}_{1}.dynspec'.format(psrname, T[0].isot) 
    psrfluxname = outdir + psrfluxname
    plotname = psrfluxname.split('.dynspec')[0]
    print(plotname)
    plt.savefig('{0}_diagnostic.png'.format(plotname))
    
    dtools.write_psrflux(dyn_clean, np.ones_like(dyn_clean), 
                           Funits, Tunits, psrfluxname, psrname=psrname,
                           note=note)
    return psrfluxname
    
if __name__ == "__main__":
    main()    
