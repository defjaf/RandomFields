# -*- coding: UTF-8 -*-
"""

check effects of pixelization and sampling.

Currently uses azimuhally symmetric pixel window functions.

TODO: check sampling effect P_noise(k) = (kσ)^2 P_sky(k) 
      check sampling effect (multiplying by Dirac comb (Shah fn)
            should check by having a version which uses the same in and out pixels
      
DONE: check mode-by-mode (kx, ky) pixel windows
      work in 1d (and 3d?) --> works automatically, except for plotting
      add in an explicit "beam" with size >> pixel (done in the notebook)
        

"""

### nb. 


from __future__ import print_function

import math
from distutils.version import LooseVersion

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.special as sps

import scipy.ndimage.filters as snf

from realization import realization
from realization import aniso
import itertools

try:
    from itertools import izip
except ImportError:
    izip = zip
    
    
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return izip(a, b)
    

def pixelWindowCircle(nu, L):
    """ 
    2-d pixel window function for [square] pixel of area L^2
    using circular window of same area
    for a circular pixel of radius R, W_circ(k, R) = 2 J_1(kR)/kR
       [J_1(x) = BesselJ(1,x)]
    area = pi R^2 so W_square(k,L) = 2 pi^(1/2) J_1[kL/pi^(1/2)]/kL
    
    nb. k = angular frequency = nu * (2π)
    """
    k = nu * 2*math.pi
    kR = k * L /np.sqrt(np.pi)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(kR<=0, 1.0, 2*sps.j1(kR)/kR)
    

def pixelWindow1D(nu, L):
    """ 
    1-d pixel window function for a 1D pixel of length L
        j_0(kL/2)
        
    should also be equivalent to:    
        2-d pixel window function for [square] pixel of area L^2
        use azimuthally symmetrized square pixel formula: j_0(kL/2)
        (not completely sure this is right)

    nb. k = angular frequency = nu * (2π)
    """
    k = nu * 2*math.pi
    kL2 = k * L/2
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(kL2<=0, 1.0, np.sin(kL2)/kL2)
        
pixelWindow2D = pixelWindowCircle ## pixelWindow1D

def pixelWindowND(nu, L, ndim=2):
    if ndim==2:
        return pixelWindow2D(nu, L)
    elif ndim==1:
        return pixelWindow1D(nu, L)
    else:
        print("ndim=%d not supported" % ndim) 
    
def pixelAverage(rlzn, width1d=2):
    """
    create a uniform-windowed field using an n-dim width1d running average in each direction.
    use scipy.ndimage.filter.uniform_filter(input, size=3, output=None, mode='reflect', cval=0.0, origin=0)
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html
    """
    
    return snf.uniform_filter(rlzn, size=width1d, mode='wrap')
    
    
def samplePower(Pk, ):
    """
    calculate the effect of sampling with spacing L on a power spectrum:
    P_L(k) = \sum_n P(|k+2π n/L|)
        start with vector k=2π m/(NL) and then average over angle m?
    """
    
    
def pixelSample(rlzn, nsamples=None, shrink_by=2):
    """ 
    create a sampled field using nsamples from a real-space field rlzn
    
    shrink the number of pixels by a factor shrink_by in each direction 
    (or should just give the output shape?)
    
    for now, just use the rlzn[...] values (could also interpolate?)
    
    """
    
    shape_in = rlzn.shape
    shape_out = tuple(np.int(s//shrink_by) for s in shape_in)
    
    print("shape_in", shape_in)
    print("shape_out", shape_out)
    print("len=%d" % rlzn.size)
    
    ## check that the the new shape is comensurate with the old shape
    assert all(s_out*shrink_by==s_in for s_in,s_out in zip(shape_in, shape_out))
    
    ### could use np.random.choice on the flattened array, 
    ###  but it's easier to track the actual array indices
    idxs_1d = np.random.choice(rlzn.size, nsamples)
    vals = rlzn.ravel()[idxs_1d]
    print("vals[%d]: %g ± %g" % (vals.size, vals.mean(), vals.std()))

    idxs_nd = np.unravel_index(idxs_1d,shape_in)
    
    idxs_nd_sampled = (np.array(idxs_nd, dtype=np.int)//shrink_by).transpose()
    rlzn_sampled = np.zeros(shape_out)
    n_in_pix = np.zeros(shape_out, np.int)
    for i, v in zip(idxs_nd_sampled, vals):
        i = tuple(i)    ## AHJ: this (occasionally?) gives non-integer i?
        rlzn_sampled[i] += v
        n_in_pix[i] += 1
        
    idxs_good = (n_in_pix>0)
    print("final shape:", idxs_good.shape)
    print("good/total: %d/%d" % (idxs_good.sum(),idxs_good.size))
    
    rlzn_sampled[idxs_good] /= n_in_pix[idxs_good] 
    print("orig: %g ± %g" % (rlzn.mean(), rlzn.std()))
    print("sampled: %g ± %g" % (rlzn_sampled.mean(), rlzn_sampled.std()))

    return rlzn_sampled
    
def unWindow(rlzn, deltas=1):
    """
    take a 2d real-space field and remove the pixel window function
    """
    sampled_delta_k = np.fft.rfftn(rlzn)
    k_values = realization.get_k(rlzn.shape, deltas)
    ndims = len(rlzn.shape)

    ### k_values.shape = (ndim, *rdims) 
    ###  -- rdims is the k-space shape (n/2), ndims is dimensionality of the space
    pixel_window_mode = np.ones_like(k_values[0,:]) 
    for d in range(ndims):
        pixel_window_mode *= pixelWindow1D(k_values[d,:], deltas)
         ### divide by single-pixel window (doesn't matter?)
        #pixel_window_mode /= pixelWindow1D(k_values[d,:], deltas) 

    sampled_delta_k_unwindowed = sampled_delta_k / pixel_window_mode
    
    return np.fft.irfftn(sampled_delta_k_unwindowed)


def subsamplePower(Pk_fun, rshape, shrink_by=2, deltas=None, nk=10, 
                   nangle=None, isFFT=False, eq_vol=False, nmax=1):
    """
    get the theoretical power spectrum of a subsampled (aliased) realization
    technically requires infinite sum
    
    P(k) = <|g(k)|^2> -> P_g(q), q=|k+2πn/L|, where k and n are vectors
    
    in terms of integers, k=2πm/(NL), so q=2π|m+nN|/(NL), m & n are integer vectors
    m->m+nN s.t. the components of m+nN<m_max = bandwidth of original signal
    """
    
    ### currently this is mostly code from aniso.getPower() -- need to modify
    
    ## can use & instead of logical_and???
        
    if nangle is None or nangle < 1:
        nangle = 1
    
    if deltas is None:
        deltas=1
        
    if isFFT:
        rshape[-1] = 2*rshape[-1]-2
        
#     fullshape = rshape[:]
    rshape_s = tuple(np.array(rshape)//shrink_by)
#     
        
    ### currently goes out to full frequency space of rshape, but really only want it to the shrunken space.    
    ndims = len(rshape)

    kvec = realization.get_k(rshape_s, deltas=deltas*shrink_by)
    
    k = np.sqrt(realization.get_k2(rshape_s, deltas=deltas*shrink_by))

    if nangle>1:
        idxsx = np.squeeze(realization.DFT_indices(rshape_s, dim1=[0], real=True))
        idxsy = np.squeeze(realization.DFT_indices(rshape_s, dim1=[1], real=True))
        ang = np.arctan2(idxsy, idxsx)    

    if eq_vol:  ## actually want uniform in k**ndims
        kk = np.linspace(0, k.max()**ndims, nk)**(1.0/ndims)
    else:
        kk = np.linspace(0, k.max(), nk)
    kout = np.zeros_like(kk)
    aa = np.linspace(0, 2*np.pi, nangle+1)
    Pk = np.zeros(shape=(nk,nangle), dtype=np.float64)

### iterating over all combinations of the -n...+n range
### see http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    ### nmaxs should really depend on where you are, such that the ktot<kmax
    ### therefore should really be inside the innermost loop below (see abs(ktot) check)
    nmaxs = (nmax,)*ndims  ### can be different in each direction
    rr = np.array([range(-nm,nm+1) for nm in nmaxs])
    kiter = np.array(np.meshgrid(*rr), dtype=np.float).T.reshape(-1, ndims)
    rfac = np.array(shrink_by, dtype=np.float)*np.array(deltas, dtype=np.float)
    kiter /= rfac   ## rescaling from raw pixels to the samples
#     print("rfac=", rfac)
#     print("kiter=", kiter)
    
    for ii,ki in enumerate(pairwise(kk)):   ### need to iterate over full k_vec in the bin as well.
        kdxi = np.logical_and(k>ki[0], k<=ki[1])
        for jj, aj in enumerate(pairwise(aa)):
            if nangle>1:
                adxj = np.logical_and(ang>aj[0], ang<=aj[1])
                idx = np.logical_and(kdxi, adxj)
            else:
                idx = kdxi
                
            ## here, idx has list pointing into power, or k
            kvec_bin = kvec[:,idx].transpose()
            powk = np.zeros(idx.sum())  ### hold the aliased power

            for ip, kv in enumerate(kvec_bin):
                for dkvec in kiter:
                    ktot = kv + dkvec
                    ### given this normalization, is the cut off at k>1/2?
                    if np.all(np.abs(ktot)<=0.5):
                        kabs = np.sqrt((ktot**2).sum())
                        pixwin = pixelWindowND(kabs, deltas*shrink_by, ndims)
                        powk[ip] += Pk_fun(2*math.pi*kabs)*pixwin**2

            ## k[idx]
            #### unwindow this bin -- just use whatever the "last" k was 
            ### -- should be more careful but OK for narrow bins
            kout[ii+1] = np.mean(ki)
            Pk[ii+1, jj] = powk.mean()/pixelWindowND(kout[ii+1], deltas*shrink_by, ndims)**2 if len(powk) else 0

    Pk = np.squeeze(Pk)
    
#     volume = (np.array(deltas)*np.array(rshape)).prod()
#     Pk /= volume
#     Sk /= volume
#     print("Volume=", volume)
    ## normalization needed for 'volume factor' 
    #      <dk dk'> = delta(k+k')P(k) => <d^2>=Vol*P(k)

#     fac = 
    
    ## or always return the same thing?
    if nangle > 1: 
        return (kout), Pk
    else:
        return kout, Pk




def samplingNoise():
    """
    extra power spectrum P_noise(k) = (kσ)^2 P_sky(k) from sampling
    note that σ^2 is the variance of the sampling locations within the pixel. 
    For uniformly distributed points within the pixel, \sigma = ...
     
    """

    
def driver(n=0, dims=(512,512), deltas=1, nsamples=1e6, shrink_by=2, 
           nk=20,allplots=False, eq_vol=False, maps=True, plot_ratio=True,
           cutraw=2, title=""):

    plot_aliased_theory=True
    
    print ('dims=', dims)
    ndims = len(dims)
    
    Pk = n
        
    deltas = 1 if deltas is None else deltas
    
    delta_k = realization.dft_realizn(dims, Pk, deltas=deltas)

    print ('delta_k: shape=', delta_k.shape)

    delta_r = np.fft.irfftn(delta_k)
    
    print ('delta_r: shape=', delta_r.shape)

    assert delta_r.shape==dims

    #### plot the map
    if ndims==2 and maps:
        plt.figure(0)
        plt.imshow(delta_r)
        plt.axis('scaled')

    if allplots and ndims==2 and maps:
        plt.figure(4)
        plt.imshow(np.abs(delta_k))
        plt.axis('scaled')
        plt.title(r'$\delta_{\bf k}$')
        
    k, P, S = aniso.getPower(delta_r, nk=nk*shrink_by, deltas=deltas, eq_vol=eq_vol)
    
    ### get the sampled field
    sampled_delta_r = pixelSample(delta_r, nsamples=nsamples, shrink_by=shrink_by)

    ### need to scale ek, eP or both by powers of shrink_by (right??)
    fac = shrink_by**(2*ndims)   ## not completely sure of this formula    
    ek, eP, eS = aniso.getPower(sampled_delta_r, nk=nk, deltas=deltas*shrink_by, 
                                eq_vol=eq_vol)
    eP *= fac; eS *= fac

    print('P[0]/eP[0]=%g, shrink**4=%g' % (P[0]/eP[0],fac))
    
    ### mode-by-mode correction for pixel window:
    sampled_delta_r_unwindowed = unWindow(sampled_delta_r, deltas=deltas*shrink_by)
    wk, wP, wS = aniso.getPower(sampled_delta_r_unwindowed, nk=nk, deltas=deltas*shrink_by,
                                eq_vol=eq_vol)
    wP *= fac; wS *= fac
    
    ### running average, full pixels
    avg_delta_r = pixelAverage(delta_r, width1d = shrink_by)
    assert avg_delta_r.shape == delta_r.shape
    ak, aP, aS = aniso.getPower(avg_delta_r, nk=nk*shrink_by, deltas=deltas, eq_vol=eq_vol)
    
    ### running average, subsampled
    sampled_avg_delta_r = avg_delta_r[[slice(None, None, shrink_by)]*ndims]

    eak, eaP, eaS = aniso.getPower(sampled_avg_delta_r, nk=nk, deltas=deltas*shrink_by, 
                                eq_vol=eq_vol)
    eaP *= fac; eaS *= fac
    assert np.all(eak == ek)

    
#     if ndims == 2:
#         plt.figure()
#         plt.imshow(delta_r)
#         plt.axis('scaled')
#         plt.figure()
#         plt.imshow(avg_delta_r)
#         plt.axis('scaled')
#         plt.figure()
#     
#     pixwin2 = pixelWindow(k,deltas)**2  ### only work for scalar deltas
#     Pw2 = P / pixwin2
#     Sw2 = S / pixwin2

    if ndims==2:
        pixelWindow = pixelWindow2D
    elif ndims==1:
        pixelWindow = pixelWindow1D
    else:
        pixelWindow = None
        
    if pixelWindow:
    
        ### sampled n_out pixels
        pixwin2 = pixelWindow(ek,deltas*shrink_by)**2  ### only work for scalar deltas
        #pixwin2 /= pixelWindow(ek,deltas)**2 ### divide by single-pixel window (doesn't matter?)
        eP_unwin= eP/pixwin2
        eS_unwin= eS/pixwin2

        eaP_unwin= eaP/pixwin2
        eaS_unwin= eaS/pixwin2
        
        ### full n_in pixels
        pixwin2a = pixelWindow(ak,deltas*shrink_by)**2  ### only work for scalar deltas
        #pixwin2a /= pixelWindow(ak,deltas)**2 ### divide by single-pixel window (doesn't matter?)
        aP_unwin= aP/pixwin2a
        aS_unwin= aS/pixwin2a
        
    if plot_aliased_theory:
        k_aliased, P_aliased = subsamplePower(Pk, dims, shrink_by=shrink_by, deltas=deltas,
                                            nk=nk*shrink_by, nmax=shrink_by, eq_vol=False)

    nraw = len(k) if not cutraw else int(len(ek)*cutraw)  ### only plot this many of the raw spectrum

    if LooseVersion(matplotlib.__version__) < LooseVersion('2'):
        colors = ['black', 'blue', 'green', 'yellow', 'orange', 'red', 'cyan', 'magenta', 'grey']
    else:
        colors = ['C'+str(n) for n in range(10)]

    plt.figure(1)
    ### k[] vector is really nu (or f) --- true k=2 pi nu. P(k) really takes
    plt.loglog(k[1:nraw], Pk(2*math.pi*k[1:nraw]), label="Theory", color=colors[0])
    
    mrkr = None ### None, '.'

    plt.loglog(k[1:nraw], P[1:nraw], label="raw P(k)", marker=mrkr, color=colors[1])
#     plt.loglog(k[1:nraw], Pw2[1:nraw], label="unwin P(k)", marker=None)
#     plt.loglog(k[1:nraw], P[1:nraw]+S[1:nraw])
    plt.xlabel(r'$k$')
    plt.ylabel(r'$P(k)$')
    
    print ('full spectra plotted. P[0]=%g' % P[0])
    
    plt.loglog(ek[1:], eP[1:], label="Poisson P(k)", marker=None, color=colors[2])
    plt.loglog(eak[1:], eaP[1:], label="sampled avg P(k)", marker=None, color=colors[3])
    #plt.loglog(wk[1:], wP[1:], label="mode-unwin Poisson P(k)", marker=None)
    plt.loglog(ak[1:nraw], aP[1:nraw], label="runavg P(k)", marker=mrkr, color=colors[4])
    plt.loglog(ak[1:nraw], aP_unwin[1:nraw], label="runavg unwin P(k)", marker=mrkr, color=colors[5])

    if pixelWindow:    
        plt.loglog(ek[1:], eP_unwin[1:], label="Poisson, unwin P(k)", marker=None, color=colors[6])
        plt.loglog(eak[1:], eaP_unwin[1:], label="sampled, avg unwin P(k)", marker=None, color=colors[7])
        
#       plt.loglog(ek[1:], eP_unwin[1:]+eS_unwin[1:])
    if plot_aliased_theory:
        plt.loglog(k_aliased[1:], P_aliased[1:], marker=None, color=colors[8], label='aliased theory')
        
    plt.legend(loc='best', frameon=False)
    plt.xlim(ek[1:].min(), ek[1:].max())
    plt.title(title)

    
    print('sampled spectra plotted, eP[0]=%g' % eP[0])
    
    if ndims==2:
        if not cutraw:
            plt.axvline(k.max()/math.sqrt(2), ls='dotted', c='grey')
        plt.axvline(ek.max()/math.sqrt(2),ls='dotted', c='grey')

    
    if plot_ratio:
        rplotter = plt.loglog ## or plt.loglog or plt.plot, ....
        Pk_theory = Pk(2*math.pi*k[1:nraw])
        plt.figure(3)
        rplotter(k[1:nraw], P[1:nraw]/Pk_theory, label="raw/theory", marker=mrkr, color=colors[1])
        rplotter(ek[1:], eP[1:]/Pk_theory[:len(ek)-1], label="Poisson/theory", marker=mrkr, color=colors[2])
        #rplotter(wk[1:], wP[1:]/Pk_theory[:len(wk)-1], label="mode-unwin/theory", marker=mrkr)
#         rplotter(ek[1:], eP[1:]/P[1:len(ek)], label="Poisson/raw", marker=mrkr)
#         rplotter(wk[1:], wP[1:]/P[1:len(ek)], label="mode-unwin/raw", marker=mrkr)
        rplotter(k[1:nraw], aP_unwin[1:nraw]/Pk_theory, label="runavg unwin/theory", marker=mrkr, color=colors[5])
        if pixelWindow:
            rplotter(ek[1:], eP_unwin[1:]/Pk_theory[:len(ek)-1], 
                        label="Poisson unwin/theory", marker=None, color=colors[6])
            rplotter(eak[1:], eaP_unwin[1:]/Pk_theory[:len(ek)-1], 
                        label="sampled, avg unwin/theory", marker=None, color=colors[7])
        if plot_aliased_theory:
            rplotter(k_aliased[1:], P_aliased[1:]/Pk(2*np.pi*k_aliased[1:]), label='aliased/theory', color=colors[8], marker=mrkr)
        plt.xlim(ek[1:].min(), ek[1:].max())
        plt.ylim(0.5, 5.0)
        plt.axhline(1.0,ls='dotted', c='grey')
        plt.legend(loc='best', frameon=False)
        plt.title(title)


    if allplots:
        #plot the power distribution
        nk1 = 9
        nktot = 81
        plt.figure(2)
        nrc = nk1-1   ## don't plot k=0
        nr = int(math.sqrt(nrc))
        nc = int(nrc/nr)
        if nr*nc<nrc: nc += 1

        for i in xrange(1,nk):
            plt.subplot(nr, nc, i)
            ang, rlzk, k = aniso.getAngDist(sampled_delta_r, i, nk=nktot)
            std = np.sqrt(((np.absolute(rlzk))**2).mean())
            plt.plot(ang, rlzk.real, '.')
            plt.plot(ang, rlzk.imag, '.')
            plt.plot([0,math.pi], [std,std], 'r')
            plt.plot([0,math.pi], [-std,-std], 'r')
            ax = plt.gca()
            lab = r'$k=%f$' % k
            plt.text(0.1, 0.9, lab, transform = ax.transAxes)

        plt.figtext(0.5, 0.93, r'$\delta(k,\theta)$')

    if ndims==2 and maps:
        plt.figure()
        plt.imshow(sampled_delta_r)
        plt.axis('scaled')
        plt.title(r'sampled $\delta({\bf r})$')
    elif ndims==1 and maps:
        plt.figure()
        lentot = len(delta_r)
        plt.plot(np.arange(lentot), delta_r, label=r'raw $\delta({\bf r})$')
        plt.plot(np.arange(0,lentot,shrink_by), sampled_delta_r, label='sampled')
        plt.legend(loc='best', frameon=False)
        plt.title(r'$\delta({\bf r}$)')



