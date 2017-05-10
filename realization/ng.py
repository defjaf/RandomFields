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

from . import realization

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
    
    ng_field_fourier = numpy.fft.rfftn(ng_field)
    
    ### anti-pattern: return depends on arguments!
    if return_config:
        return (ng_field_fourier, ng_field)
    else:
        return ng_field_fourier
        
        
    
    
    
    


