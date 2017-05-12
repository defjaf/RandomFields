""" 

make a realization of a non-gaussian field (with PDF TBD)

Examine its real- and fourier-space statistics

Matsubara 2006 says that the 1-point PDF of fourier modes should be 
Gaussian as V->\infty (which I think therefore means for scales much
smaller than the box).

"""


from __future__ import division
from __future__ import with_statement

import math
import numbers

import numpy as np
import numpy.random as Nr

import scipy.stats as sps

from . import realization
from . import aniso.pairwise as pairwise

def local_fNL(dims, fNL, Pk=None, deltas=None, return_config=False):
    """
    make a realization of local fNL non-Gaussianity: 
        f = g + fNL(g^2 - <g^2>)
    where g is a Gaussian field. 
    
    dims, Pk, deltas are as in 
        realization.dft_realizn(dims, Pk=None, deltas=None)
    """
    
    ### have to go into config space, so return that, or back to fourier?
    
    gaussian_field_fourier = realization.dft_realizn(dims, Pk=Pk, deltas=deltas)
    gaussian_field = numpy.fft.irfftn(gaussian_field_fourier)
    
    gaussian_variance = np.variance(gaussian_field)
    
    ng_field = gaussian_field.copy()
    ng_field -= fNL*(gaussian_field**2 - gaussian_variance)
    
    ng_field_fourier = np.fft.rfftn(ng_field)
    
    ### anti-pattern: return depends on arguments!
    if return_config:
        return (ng_field_fourier, ng_field)
    else:
        return ng_field_fourier
        

def get_fourier_dist(rlzn, deltas=None, nk=10, nbins=30, isFFT=True, eq_vol=False):):
    """
    get the distribution (histogram and summary statistics)
    of fourier components in nk bands of k, nbins histogram bins
    
    
    """
    
    #TODO:
    ## normalise to variance (power) in the bin?
    ## return numpy hist output; plot elsewhere?
    ## require same hist binning in all bands? or optimize?
    
    ### code below from aniso.getPower
    
    if deltas is None:
        deltas=1
        
    rshape = rlzn.shape
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
    
    ## just use lists so we can just append
    stats = []
    hists = []
    
    for ki in pairwise(kk):
        kdxi = np.logical_and(k>ki[0], k<=ki[1])
        rlzn_bin = rlzn[kdxi]
        full_bin = np.hstack((rlzn.re(),rlzn.im())
        if normalized
            full_bin = (full_bin - full_bin.mean())/full_bin.std()
        
        stati = sps.describe(full_bin)[2:]  #mean, var, skew, kurt
        histi = np.histogram(full_bin, bins=nbins)
        
        stats.append(stati)
        hists.append(histi)
        
        return stats, hists

