import numpy as np
import astropy.units as u
import argparse
from astropy.io import fits
from astropy.time import Time
from astropy.constants import c

from scipy.ndimage.filters import gaussian_filter, median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scintillation.dynspectools.slowft as slowft
import scintillation.dynspectools.dynspectools as dtools
import scipy.optimize as so

def plot_secspec(dynspec, freqs, t0, xlim=None, ylim=None, bintau=2, dt=4*u.s, vm=3., sft=True,
                 binft=1, aspect=(10,10), plot_title=False, curv=0, normS=0, half=1, psrname=' '):

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
    
    if normS:
        Ngrid = 1000
        taumin=0.2
        Sb_norm = np.copy(Sb)
        Sb_norm[abs(ft)<2.] = 0
        Snorm, ftnormaxis, curvaxis = norm_sspec(Sb_norm, ftb, tau, Ngrid=Ngrid, taumin=taumin)

    
    slow = np.median(Sb)-0.2
    shigh = slow + vm

    # Not the nicest, have a set of different plots it can produce
    plt.figure(figsize=aspect)
    if normS:
        ax2 = plt.subplot2grid((5, 2), (0, 0), rowspan=5)
        ax3 = plt.subplot2grid((5, 2), (3, 1), rowspan=2)
        ax4 = plt.subplot2grid((5, 2), (1, 1), rowspan=2)
        axprof = plt.subplot2grid((5, 2), (0, 1), rowspan=1)
    else:
        ax2 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
        ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

    plt.subplots_adjust(wspace=0.1, hspace=0.02)

    # Plot dynamic spectrum image

    ax2.imshow(dspec_plot.T, aspect='auto', vmax=mn+5, vmin=mn-2, origin='lower',
                extent=[0,T,min(freqs), max(freqs)], cmap='viridis')
    ax2.set_xlabel('time (min)', fontsize=16)
    ax2.set_ylabel('freq (MHz)', fontsize=16)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")

    # Plot Secondary spectrum
    ax3.imshow(Sb.T, aspect='auto', vmin=slow, vmax=shigh, origin='lower',
               extent=[min(ft), max(ft), min(tau), max(tau)], interpolation='nearest',
              cmap='viridis')
    ax3.set_xlabel(r'$f_{D}$ (mHz)', fontsize=16)
    ax3.set_ylabel(r'$\tau$ ($\mu$s)', fontsize=16) 

    if curv:
        plt.plot(ft, curv*ft**2.0, linestyle='dotted', color='tab:orange')
    
    if normS:
        logimg = np.copy(Snorm)
        logimg[abs(logimg)< 1e-9] = 0 #np.nan
        mask = np.zeros_like(logimg)
        mask[~np.isnan(logimg)] = 1

        xgrid = np.copy(ftnormaxis)

        taulow = 0.2
        taumax = max(tau)
        xlim = max(ftnormaxis)

        powerprof = np.nanmean(logimg[:,(tau>taulow)], axis=-1)
        ic = np.nanmean(mask[:,(tau>taulow)], axis=-1)
        ic[ic<1e-3] = 1
        powerprof = powerprof / ic

        xsmooth = 0
        dxgrid = xgrid[1] - xgrid[0]
        xbin_smooth = int(xsmooth/dxgrid)
        if xsmooth:
            powerprof = powerprof - gaussian_filter(powerprof, xbin_smooth)

        ax4.imshow(Snorm.T, aspect='auto', vmin=slow, vmax=shigh, origin='lower',
               extent=[min(ft), max(ft), min(tau), max(tau)], interpolation='nearest',
              cmap='viridis')
        ax4.set_xticks([])
        ax4.set_ylim(taulow, max(tau))
        ax4.yaxis.tick_right()
        ax3.set_xlabel(r'$f_{D}$ (mHz)', fontsize=16)
        ax3.set_ylabel(r'$\tau$ ($\mu$s)', fontsize=16)
        axprof.set_xticks([])
        axprof.plot(ftnormaxis, powerprof)
        axprof.yaxis.tick_right()
        axprof.yaxis.set_label_position("right")

    if xlim:
        ax3.set_xlim(-xlim, xlim)
    if normS:
        ax3.set_ylim(0, max(tau))
    else:
        ax3.set_ylim(min(tau), max(tau))
    if plot_title:
        ax3.set_title(' {0}'.format(psrname), fontsize=18)
        ax2.set_title(' {0}'.format(t0.isot), fontsize=18)
        
    if normS:
        return CS, ft, tau, Snorm, ftnormaxis, curvaxis
    else:
        return CS, ft, tau


def norm_sspec(S, ft, tau, Ngrid=2000, taumin=0.1, tauref=None, xr=1.):
    """
    Create normalized secondary spectrum, as in Reardon et al. 2020

    Currently does not take into account flux preservation, it simply creates
    a scaled ft axis, and inserts the closest value of S in each
    """
    
    img = np.zeros( (Ngrid+1, len(tau) ) )
    if not tauref:
        tauref = max(tau)
        
    xgrid = np.linspace(-xr*max(ft), xr*max(ft), Ngrid+1, endpoint=True)    
    for i in range(len(tau)):
        taui = tau[i]
        if (taui > taumin):
            for j in range(Ngrid+1):
                fti = xgrid[j]*np.sqrt(taui/ tauref)
                findex = find_nearest(ft, fti)
                img[j, i] = S[findex, i]
                
    curvaxis = tauref / (xgrid)**2.0
    return img, xgrid, curvaxis
                   
    
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def tau_acf_model(x, A, dt):
    """
    Fit 1D function to cut through ACF for scintillation timescale.
    Exponent is 5/3 for Kolmogorov turbulence.
        amp = Amplitude
        tau = timescale at 1/e
        alpha = index of exponential function. 2 is Gaussian, 5/3 is Kolmogorov
    """

    bw = max(x) - min(x)
    
    model = A*np.exp(-(np.abs(x)/dt)**(5./3))
    # Multiply by triangle function
    model = np.multiply(model, 1-(np.abs(x)/bw) )
    return model


def dnu_acf_model(x, A, dnu):
    """
    Fit 1D function to cut through ACF for decorrelation bandwidth.
    Default function has is exponential with dnu measured at half power
        amp = Amplitude
        dnu = bandwidth at 1/2 power
    """

    tobs = max(x) - min(x)

    model = A*np.exp(-(np.log(2)*np.abs(x)/dnu) )
    # Multiply by triangle function
    model = np.multiply(model, 1-(np.abs(x)/tobs) )
    return model

def scint_acf_model_2d_approx(xy, A, dt, dnu, phasegrad):
    """
    Fit an approximate 2D ACF function
    
    Function skeleton taken from scintools
    """

    ### Hardcoded fref to 1 GHz, fix in future
    fref = 1000.
    alpha = 5./3
    
    dt_axis = xy[0][0]
    df_axis = xy[1][:,0]
    
    bw = max(df_axis) - min(df_axis)
    tobs = max(dt_axis) - min(dt_axis)
    
    nt = len(dt_axis)
    nf = len(df_axis)

    tdata = np.reshape(dt_axis, (nt, 1))
    fdata = np.reshape(df_axis, (1, nf))

    model = A * np.exp(-(abs((tdata / dt) + 2 * phasegrad *
                          ((dnu / np.log(2)) / fref)**(1./6) *
                          (fdata / (dnu / np.log(2))))**(3. * alpha / 2) +
                       abs(fdata / (dnu / np.log(2)))**(3./2))**(2./3))

    # multiply by triangle function
    model = np.multiply(model, 1-np.divide(abs(tdata), tobs))
    model = np.multiply(model, 1-np.divide(abs(fdata), bw))
    model = np.fft.fftshift(model)
    model = np.fft.ifftshift(model)
    #model = np.transpose(model)

    return model.ravel()

def corr2D(dynspec, dt, df):
    
    dynpad = np.pad(dynspec, pad_width=((dynspec.shape[0], 0), (0, dynspec.shape[1])), 
                mode='constant', #constant_values=0)
                    constant_values=np.mean(dynspec)) 
    
    ft = np.fft.rfft2(dynpad)
    corr = np.fft.irfft2(ft * np.conj(ft))
    
    ft = np.fft.fftfreq(dynpad.shape[0], dt)
    ft = np.fft.fftshift(ft.to(u.mHz).value)
    
    tau = np.fft.fftfreq(dynpad.shape[1], df)
    tau = np.fft.fftshift(tau.to(u.microsecond).value)
    
    df_axis = np.fft.fftfreq(dynpad.shape[1], d=(tau[1]-tau[0]) )
    dt_axis = np.fft.fftfreq(dynpad.shape[0], d=(ft[1]-ft[0])*u.mHz ).to(u.min).value
    
    return corr, dt_axis, df_axis

def fitCorr(dynspec, F, dt=8*u.s, fit="1D", aspect=(6,6), plot_xlabel=0):
    """
    Perform 2D ACF, fit with 2D gaussian
    Additionally fit t, f slices with 1D gaussians
    
    """
    
    taxis = np.arange(dynspec.shape[0])*dt
    dyn_chunk = np.copy(dynspec)
    dF = (F[1] - F[0])*u.MHz
    corr, dt_axis, df_axis = corr2D(dyn_chunk, dt, dF)
    
    # Fill center spike with extrapolation of ccorr
    dcorr = np.mean(np.diff(corr[1:6]))
    corr[0] = (corr[1] + corr[-1])/2. - dcorr
    
    midf = corr.shape[-1]//2
    midt = corr.shape[0]//2

    corrplot = np.fft.fftshift(corr)
    corrplot -= np.median(corrplot)
    corrplot /= np.max(corrplot)

    df_shifted = np.fft.fftshift(df_axis)[:corrplot.shape[1]]
    dt_shifted = np.fft.fftshift(dt_axis)
        
    tp, nup = np.meshgrid(dt_shifted,df_shifted)
    z = [tp, nup]
    data = corrplot.ravel()
        
    #ccorr_f = (corrplot[midt-1] + corrplot[midt+1])/2.
    #ccorr_t = (corrplot[:,midf-1] + corrplot[:,midf+1])/2.
    ccorr_f = corrplot[midt]
    ccorr_t = corrplot[:,midf]
        
    # 1D fits
    tfit, tfiterr = so.curve_fit(tau_acf_model, dt_shifted, ccorr_t, p0=[1.,5.], maxfev=10000)
    nufit, nufiterr = so.curve_fit(dnu_acf_model, df_shifted, ccorr_f, p0=[1.,5.], maxfev=10000)

    tmodel = tau_acf_model(dt_shifted, *tfit)
    fmodel = dnu_acf_model(df_shifted, *nufit)

    if fit == '2D':
        p0 = [1, tfit[1], nufit[1], 0]

        b, berr = so.curve_fit(scint_acf_model_2d_approx, z, data, p0=p0, 
                               maxfev=50000, bounds=( [0,0,0,-10.], 
                                       [np.inf,np.inf,np.inf,10.]) )
    
        model = scint_acf_model_2d_approx(z, *b)
        model = model.reshape(corrplot.shape)

        xwidth = b[1]
        ywidth = b[2]
        ang = b[3]
        angerr = np.sqrt(berr[3,3])    
        xerr = np.sqrt(berr[1,1])    
        yerr = np.sqrt(berr[2,2]) 
        
    elif fit == "1D":
        xwidth = tfit[1]
        xerr = np.sqrt(tfiterr[1,1])
        ywidth = nufit[1]
        yerr = np.sqrt(nufiterr[1,1])
        ang = -1
        angerr = -1
        
    else:
        print("Fit type must be 1D or 2D")
    
    plt.figure(figsize=aspect)

    if fit=="2D":

        ax1 = plt.subplot2grid((4, 4), (1, 0), colspan=3, rowspan=3)
        ax2 = plt.subplot2grid((4, 4), (1, 3), rowspan=3)
        ax3 = plt.subplot2grid((4, 4), (0, 0), colspan=3)

        plt.subplots_adjust(wspace=0.05)

        ax1.imshow(corrplot.T, aspect='auto', origin='lower',
                  extent=[min(dt_axis), max(dt_axis), min(df_axis), max(df_axis)])
        ax1.set_xlabel('dt (min)', fontsize=16)
        ax1.set_ylabel(r'd$\nu$ (MHz)', fontsize=16)

        ax2.plot( ccorr_f, df_shifted, color='k')
        ax2.plot( fmodel, df_shifted, color='tab:orange', linestyle='dotted')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.set_xlabel(r'I (d$\nu$, dt=0)', fontsize=16)
        ax2.set_ylim(min(df_axis), max(df_axis) )

        ax3.plot( dt_shifted, ccorr_t, color='k')
        ax3.plot( dt_shifted, tmodel, color='tab:orange', linestyle='dotted')
        ax3.set_ylabel(r'I (dt, d$\nu$=0)', fontsize=16)
        ax3.set_xlim(min(dt_axis), max(dt_axis))
        ax3.set_xticks([])

        # 2D model contours and cuts
        ax1.contour(dt_shifted, df_shifted, model.T, 3, cmap='hot', alpha=0.5)
        ax2.plot( model[midt], df_shifted, color='tab:blue', linestyle='dotted')
        ax3.plot( dt_shifted, model[:,midf], color='tab:blue', linestyle='dotted')

    if fit=="1D":

        ax2 = plt.subplot(121)
        ax3 = plt.subplot(122)

        plt.subplots_adjust(hspace=0.05)

        ax2.plot( df_shifted, ccorr_f, color='k')
        ax2.plot( df_shifted, fmodel, color='tab:orange', linestyle='dotted')
        ax2.set_ylabel(r'I (d$\nu$, dt=0)', fontsize=14)
        ax2.set_xlim(min(df_axis), max(df_axis) )

        ax3.plot( dt_shifted, ccorr_t, color='k')
        ax3.plot( dt_shifted, tmodel, color='tab:orange', linestyle='dotted')
        ax3.set_ylabel(r'I (dt, d$\nu$=0)', fontsize=14)
        ax3.yaxis.set_label_position("right")
        ax3.yaxis.tick_right()
        ax3.set_xlim(min(dt_axis), max(dt_axis))
        if plot_xlabel:
            ax2.set_xlabel(r'd$\nu$ (MHz)', fontsize=14)
            ax3.set_xlabel('dt (min)', fontsize=14)
        else:
            ax2.set_xticks([])
            ax3.set_xticks([])

    return xwidth, xerr, ywidth, yerr, ang, angerr, z

def plot_scintpars(Fmid, tscints, tscinterrs, nuscints, nuscinterrs):

    plt.figure(figsize=(6,3))

    ax1 = plt.subplot(111)
    ax2 = ax1.twinx()

    ax1.errorbar(Fmid, tscints, yerr=tscinterrs, linestyle='none', 
                 marker='o', color='tab:blue', alpha=0.5)
    ax2.errorbar(Fmid, nuscints, yerr=nuscinterrs, linestyle='none',
                 marker='*', color='tab:orange', alpha=0.5)

    ax1.tick_params(axis='y', colors='tab:blue')
    ax2.tick_params(axis='y', colors='tab:orange')
    ax1.set_ylim(0, max(tscints)*1.5)
    ax2.set_ylim(0, max(nuscints)*1.2)
    ax1.set_xticks(Fmid)

    ax1.set_xlabel('Frequency (MHz)', fontsize=14)
    ax1.set_ylabel(r'$t_{\rm scint}$ (min)', fontsize=14, color='tab:blue')
    ax2.set_ylabel(r'$\nu_{\rm scint}$ (MHz)', fontsize=14, color='tab:orange')


def fit_parabola(x, curv, x0=0, C=0):
    return curv*(x-x0)**2 + C

def fit_curvature(Snorm, ftnormaxis, tauaxis, taulow=0.2, ftcut=2, plot=0):
    """
    Fit for arc curvature from normalized secondary spectrum
    
    Values hardcoded for use on Meerkat TPA dataset
    """
    
    logimg = np.copy(Snorm)
    logimg[abs(logimg)< 1e-9] = 0
    
    mask = np.zeros_like(logimg)
    mask[~np.isnan(logimg)] = 1

    for i in range(logimg.shape[1]):
        zrange = np.argwhere(logimg[:,i]==0).squeeze()
        z0 = max(min(zrange)-1, 0)
        z1 = min(max(zrange)+1, logimg.shape[0]-1)
        logimg[zrange,i] = (logimg[z0, i] + logimg[z1, i])/2.
        
    xgrid = np.copy(ftnormaxis)

    taumax = max(tauaxis)
    xlim = max(ftnormaxis)

    powerprof = np.nanmean(logimg[:,(tauaxis>taulow)], axis=-1)
    ic = np.nanmean(mask[:,(tauaxis>taulow)], axis=-1)
    ic[ic<1e-3] = 1
    powerprof = powerprof / ic

    xsmooth = 5
    dxgrid = xgrid[1] - xgrid[0]
    xbin_smooth = int(xsmooth/dxgrid)
    # subtract smoothed profile from power prof
    if xsmooth:
        powerprof = powerprof - gaussian_filter(powerprof, xbin_smooth)

    # average over +ve and -ve ftaxis
    midprof = powerprof.shape[0]//2
    curvprof = powerprof[:midprof][::-1] + powerprof[midprof+1:]
    #curvax2 = curvaxis[midprof+1:]
    ftprofaxis = ftnormaxis[midprof+1:]

    # Take peak as starting guess
    ftmax_index = np.argmax(curvprof)
    ft0 = ftprofaxis[ftmax_index]
    C0 = curvprof[ftmax_index]
    ftlim = 20

    fitrange = slice(ftmax_index-ftlim, ftmax_index+ftlim+1)
    ft_fit = ftprofaxis[fitrange]
    prof_fit = curvprof[fitrange]

    p0 = [-0.1, ft0, C0]
    popt, pcov = so.curve_fit(fit_parabola, ft_fit, prof_fit, p0=p0, )

    tauref = taumax
    ftcurv = popt[1]
    ftcurverr = np.sqrt(pcov[1][1])

    curvavg = tauref / ftcurv**2.0
    curvmin = tauref / (ftcurv+ftcurverr)**2
    curvmax = tauref / (ftcurv-ftcurverr)**2
    curverr = (abs(curvavg-curvmax) + abs(curvavg-curvmin))/2.

    if plot:
        plt.figure(figsize=(5,3))
        plt.plot(ftprofaxis, curvprof)
        plt.plot(ft_fit, fit_parabola(ft_fit, *popt))
        plt.xlabel(r'$f_{D}$ (mHz)', fontsize=14)
        plt.ylabel('log10 power (arb)', fontsize=14)
        
    return curvavg, curverr, ftcurv, ftcurverr
    

def main(raw_args=None):

    parser = argparse.ArgumentParser(description='temp description')
    parser.add_argument("-fname", type=str)
    parser.add_argument("-outdir", type=str)
    
    a = parser.parse_args(raw_args)
    fn = a.fname
    outdir = a.outdir
    prefix = outdir + (fn.split('/')[-1]).split('.dynspec')[0]
    res = 200
    
    ds_filtered, ds_err, T, Funits, psrname = dtools.read_psrflux(fn)        
    t0 = Time(T[0], format='mjd')
    F = Funits.value
    
    # crop all to 928 freq channels, hardcoded to Meerkat TPA
    if ds_filtered.shape[1] == 1024:
        ds_filtered = ds_filtered[:, 48:-48]
        F = F[48:-48]
    fref = np.mean(F)
    
    Tunits = Time(T)
    Tunits = (Tunits.unix - Tunits[0].unix)*u.s
    dt = Tunits[2] - Tunits[1]

    # Plot secspec using full band
    CS, ft, tau = plot_secspec(ds_filtered, F, t0, dt=dt, vm=4., sft=False, aspect=(10,10),
                               plot_title=True, psrname=psrname)
    plt.tight_layout()
    plt.savefig('{0}_fulldyn.png'.format(prefix), dpi=res)
    
    # Average out smooth response for use in ACF
    bpass = ds_filtered.mean(0)
    bpass_smooth = gaussian_filter(bpass, 53)
    
    # two clean bands
    frange1 = np.argwhere((F>970) & (F<1076)).squeeze()
    frange2 = np.argwhere((F>1291) & (F<1529)).squeeze()
    
    # In each of two clean bands, compute secondary spectra, 2D ACF, arc curv
    fitpars_2sub = np.zeros((2, 9))
    j = 0
    for frange in [frange1, frange2]:
        Fmed = np.mean(F[frange])
        #curv = normcurv * (Fmed/fref)**-2.0
        CS, ft, tau = plot_secspec(
                        ds_filtered[:,frange], F[frange], t0, dt=dt, 
                        bintau=1, binft=1, vm=4., sft=False, 
                        aspect=(8,5), normS=0) #curv=curv
        plt.tight_layout()
        plt.savefig('{0}_dyn_{1}.png'.format(prefix, int(Fmed)), dpi=res)
    
        #tscint, tscinterr, nuscint, nuscinterr, grad, graderr, maxes = fitCorr(
        #    ds_filtered[:,frange]/bpass_smooth[frange], F[frange], fit='2D')
        #plt.tight_layout()
        #plt.savefig('{0}_2Dacf_{1}.png'.format(prefix, int(Fmed)), dpi=res)
        
        #try:
        #    curvavg, curverr, ftcurv, ftcurverr = fit_curvature(Snorm, ftnormaxis, tau, plot=1)
        #    plot.tight_layout()
        #    plt.savefig('{0}_arcfit_{1}.png'.format(prefix, int(Fmed)), dpi=res)
        #except:
        #    print("Curvature fitting failed")
        #    curvavg = -1
        #    curverr = -1
        #fitpars_2sub[j] = np.array([Fmed, tscint, tscinterr, nuscint, nuscinterr, grad, graderr,
        #                   curvavg, curverr])
        #j += 1
            
    # Compute scint values in 8 frequency  slices
    #tscints = np.zeros(8)
    #nuscints = np.zeros(8)
    #tscinterrs = np.zeros(8)
    #nuscinterrs = np.zeros(8)
    #modindeces = np.zeros(8)
    #Fmid = np.zeros(8)
    
    ### Supplement nuscint with modindex
    
    #for k in range(8):
    #    frange = slice( (8-k-1)*118, (8-k)*118)
    #    dynslice = ds_filtered[:,frange]/bpass_smooth[frange]
    #    if k == 7:
    #        plot_xlabel = 1
    #    else:
    #        plot_xlabel = 0
    #    tscint, tscinterr, nuscint, nuscinterr, grad, graderr, maxes = fitCorr(
    #        dynslice, F[frange], fit='1D', aspect=(6,1.4), plot_xlabel=plot_xlabel)
    
    #    tscints[k] = tscint
    #    tscinterrs[k] = tscinterr
    #    nuscints[k] = nuscint
    #    nuscinterrs[k] = nuscinterr
    #    Fmid[k] = np.mean(F[frange])
    #    modindeces[k] = np.std(dynslice) / np.mean(dynslice)
    
    #    tscint = np.round(tscint, 2)
    #    tscinterr = np.round(tscinterr, 2)
    #    nuscint = np.round(nuscint, 2)
    #    nuscinterr = np.round(nuscinterr, 2)
    #    print("dt={0}+-{1}min, df={2}+-{3}MHz".format(tscint, tscinterr,
    #                                              nuscint, nuscinterr))
        #plt.tight_layout()
    #    if k == 7:
    #        plt.gcf().subplots_adjust(bottom=0.2)
    #    plt.savefig('{0}_1Dacf_{1}.png'.format(prefix, k), dpi=res)
        
    #plot_scintpars(Fmid, tscints, tscinterrs, nuscints, nuscinterrs)
    #plt.gcf().subplots_adjust(bottom=0.2)
    #plt.savefig('{0}_scintpars.png'.format(prefix), dpi=res)
    
    #plt.figure(figsize=(6,6))
    #plt.plot()
    #plt.xlim(0,1)
    #plt.ylim(0,1)
    

    # Print table of values
    #fitpars_print = np.around(fitpars_2sub, 3)
    
    #for i in range(2):
    #    offset = 0.5*i
    #    plt.text(0.05, 0.42+offset, "F = {0} MHz".format(fitpars_print[i,0]), fontsize=16)
    #    plt.text(0.05, 0.33+offset, r"$ t_s = {0}\pm{1}$ min".format(fitpars_print[i, 1],
    #                                                                 fitpars_print[i, 2] ), fontsize=14)
    #    plt.text(0.05, 0.27+offset, r"$ \nu_s = {0}\pm{1}$ MHz".format(fitpars_print[i, 3],
    #                                                                   fitpars_print[i, 4] ), fontsize=14)
    #    plt.text(0.05, 0.20+offset, r"$ tilt = {0}\pm{1}$ deg".format(fitpars_print[i, 5],
    #                                                                  fitpars_print[i, 6] ), fontsize=14)
    #    plt.text(0.05, 0.13+offset, r"$ \eta = {0}\pm{1}$ s^3".format(fitpars_print[i, 7],
    #                                                                  fitpars_print[i, 8] ), fontsize=14)
    #    plt.axhline(0.5, color='k')
    #    plt.xticks([])
    #    plt.yticks([])
    #    plt.savefig('{0}_table.png'.format(prefix), dpi=res)
    
    #with open('{0}_fitpars.txt'.format(prefix), 'w') as outfile:
    #    outfile.write("F, tscint, tscinterr, nuscint, nuscinterr, ACFtilt, ACFtilterr, curv, curverr \n")
    #    outfile.write("{0} \n".format(fitpars_2sub[0]))
    #    outfile.write("{0} \n".format(fitpars_2sub[1]))

    return prefix

if __name__ == "__main__":
    main()    
