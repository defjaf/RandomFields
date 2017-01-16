"""
make a realization of an 'ill-proportioned' toroidal universe with
given P(k), excise a small cubic box, and examine the vector power
spectrum P(|k|, angle)

"""
from __future__ import division

import math
import copy

import numpy as N
import numpy.random as Nr
import pylab
import scipy

def dft_realizn(n, Pk=None, scale=1):
    """
    make a realization of f(k) with
      n: sequence giving the number of frequency elements in each dimension
         so number of dimensions = size(n)
         
      Pk: power spectrum [vector or function?]

      scale: scale factors for converting integers k[i] into spatial
             frequencies (reciprocal? scale factors for lengths?)

      nb: last dimension of the output is length n[-1]/2 since f(r) is real

    current version doesn't bother explicitly filling the f(k) array.
    Instead, makes a realization of real white noise which is then FFTed to give w(k)
    Then f(k) = sqrt[P(k)]*w(k)
    """

    nn = N.array(n)
    ntot = N.prod(nn)
    
    wr = Nr.normal(0,1,size=ntot)  ## white noise
    wr.shape = nn
    wk = N.dft.real_fftnd(wr)      

    if Pk is None:
        return wk
    else:
        return wk*N.sqrt(Pk(N.sqrt(get_k2(n, scale=scale))))
    

def excise(rlzn, n, start=0, scale=1, isFourier=False):
    """
    excise a region of dimensions n[...] from the realization rlzn.

    if start is specified, 
    if (isFourier): realization is in fourier space, otherwise real space.
    """
    pass

def getPower(rlzn):
    """
    get the power spectrum of the realization

    plot here, too?
    """
    pass

def DFT_indices(dimensions, dtype=N.int_, dim1=None, real=False):
    """DFT_indices(dimensions,dtype=int_) returns an array representing a grid
    of DFT indices with row-only, and column-only variation.
    A DFT index is defined so that they go (0, 1, 2, .... N/2, -(N/2-1), .... -1)
    (see numpy.dft.fft)

    for a real DFT, the final index only runs 0...N/2
       (nb., in that case dimensions=the full dimensionality in configuration space)

    if dim1 is specified, only return the indices along a single dimension

    """

    dims = list(dimensions)
    ndims = len(dims)
    if real: dims[-1]=dims[-1]/2+1
    tmp = N.ones(dims, dtype)
    lst = []

    if dim1 is None:
        dim1 = range(ndims)

    for i in dim1:
        d = dims[i]
        fidx = N.add.accumulate(tmp, i, )-1

        if not (real and i==ndims-1):
            fidx = N.where(fidx<=d/2, fidx, fidx-d)
        lst.append(fidx)

    return N.array(lst)


def get_k2(dims, scale=None):
    """ return an array giving the value of |k|^2 for a real DFT matrix

        nb. dims are the full dimensionality in configuration space
    """

    rdims = list(dims)
    rdims[-1]=rdims[-1]/2+1
    k2 = N.zeros(shape=rdims, dtype=N.float64)
    for i in range(len(dims)):
        idx = DFT_indices(dims, dim1=[i], real=True)   ### get only a single dimension
        if scale is None:
            k2 += idx[0]**2
        else:
            try:
                scale_i = scale[i]
            except TypeError:
                scale_i = scale            
            k2 += (idx[0]**2)/(scale_i**2)
            
    return k2

def driver(n=0, dims=(512,512), scale=None):

    print 'dims=', dims

    if n<0:
        def Pk(k):
            return N.where(k<=0, 0, k**n)
    else:
        def Pk(k):
            return k**n
    
    delta_k = dft_realizn(dims, Pk, scale=scale)

    print 'delta_k: shape=', delta_k.shape

    delta_r = N.dft.inverse_real_fft2d(delta_k)

    print 'delta_r: shape=', delta_r.shape

    assert delta_r.shape==dims

    pylab.imshow(delta_r)
