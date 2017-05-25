""" 

make a realization of a non-gaussian field (with PDF TBD)

Examine its real- and fourier-space statistics

Matsubara 2006 says that the 1-point PDF of fourier modes should be 
Gaussian as V->\infty (which I think therefore means for scales much
smaller than the box).

"""


from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import math
import numbers

import numpy as np
import numpy.random as Nr

import scipy.stats as sps

import realization
from aniso import pairwise

def local_fNL(dims, fNL, Pk=None, deltas=None, ng_pow=2, return_config=False, fG=1.0, transform=None):
    """
    make a realization of local fNL non-Gaussianity: 
        f = fG*g + fNL(g^p - <g^p>)
    where g is a Gaussian field and p is a power. If p<0, ignore the offset term.
    
    or, supply a function transform(g)
    
    dims, Pk, deltas are as in 
        realization.dft_realizn(dims, Pk=None, deltas=None)
    """
    
    ### have to go into config space, so return that, or back to fourier?
    
    gaussian_field_fourier = realization.dft_realizn(dims, Pk=Pk, deltas=deltas)
    gaussian_field = np.fft.irfftn(gaussian_field_fourier)
        
    if transform is not None:
        try:
            ng_field = transform(gaussian_field.copy())
        except NameError:
            print("Must supply a callable transform")
            raise
    
    if transform is None:
        gaussian_offset = np.mean(gaussian_field**ng_pow) if ng_pow>0 else 0.0

        ng_field = fG*gaussian_field.copy()
        ng_field += fNL*(gaussian_field**ng_pow - gaussian_offset)
        
    
    ng_field_fourier = np.fft.rfftn(ng_field)
    
    ### anti-pattern: return depends on arguments!
    if return_config:
        return (ng_field_fourier, ng_field)
    else:
        return ng_field_fourier
        

def get_fourier_dist(rlzn, deltas=None, nk=10, nbins=30, 
                     isFFT=True, eq_vol=False, normalized=True):
    """
    get the distribution (histogram and summary statistics)
    of fourier components in nk bands of k, nbins histogram bins
    
    """
    
    #TODO:
    ## normalise to variance (power) in the bin?
    ## return numpy hist output; plot elsewhere?
    ## require same hist binning in all bands? or optimize?
    
    ### code below from aniso.getPower
        
    if deltas is None:
        deltas=1.0
        
    rshape = list(rlzn.shape)
    if isFFT:
        rshape[-1] = 2*rshape[-1]-2
        
    ndims = len(rshape)

    k = np.sqrt(realization.get_k2(rshape, deltas=deltas))
    
    if eq_vol:  ## actually want uniform in k**ndims
        kk = np.linspace(0, k.max()**ndims, nk)**(1.0/ndims)
    else:
        kk = np.linspace(0, k.max(), nk)

    kout = np.zeros_like(kk)

    ## fourier-space realization is complex: combine real and imag parts?
    ## how to deal with real-valued diagonal entries/mean 0?
    
    ## just use lists so we can just append
    stats = []
    hists = []
    kout = []
    
    for ki in pairwise(kk):
        kdxi = np.logical_and(k>ki[0], k<=ki[1])
        kbin = np.average(ki)   
            ### or average over the actual k that contribute?
        
        rlzn_bin = rlzn[kdxi]
        full_bin = np.hstack((rlzn_bin.real,rlzn_bin.imag)).ravel()
        if normalized:
            full_bin = (full_bin - full_bin.mean())/full_bin.std()
        
        stati = sps.describe(full_bin, axis=None)[2:]  #mean, var, skew, kurt
        histi = np.histogram(full_bin, bins=nbins)
        
        stats.append(stati)
        hists.append(histi)
        kout.append(kbin)
        
    return np.array(kout), np.array(stats), hists


def driver(dims = (256,256,256),
            fNL = 1000.0,
            deltas = None,
            nk = 10,
            nbins = 30,
            eq_vol=False,
            Pk = -2,
            ng_pow=2, fG=1.0,
            transform=None):

    
    rlzn = local_fNL(dims, fNL, 
                     Pk=Pk, deltas=deltas, return_config=False, ng_pow=ng_pow, 
                     fG=fG, transform=transform)
    
    kout, stats, hists = get_fourier_dist(rlzn, deltas=deltas, nk=nk, nbins=nbins, 
                     isFFT=True, eq_vol=eq_vol, normalized=True)
                     
    return kout, stats, hists
    