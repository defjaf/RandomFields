"""
make a realization of a field with given P(k)

arbitrary number of dimensions and number of points in each

"""

#### need to make into a constant Delta f=i/Delta
#### maybe make a class that carries along delta with it?

from __future__ import division
from __future__ import with_statement

import math
import numbers

import numpy as np
import numpy.random as npr

# from operator import isNumberType

def dft_realizn(dims, Pk=None, deltas=None):
    """
    make a realization of f(k) with
      dims: sequence giving the number of frequency elements in each dimension
            so number of dimensions = size(dims)
      
      Pk: power spectrum
         if a function, then just calculate Pk(k)
         if a number, then return k**n for n=Pk
                (return 0 for k=0 if n<0)
         if None, then return white noise (n=0)

      
      deltas: scale factors for converting integers k[i] into spatial
             frequencies: kphys[]=k[]/(deltas[i]*dims[i]) for [] in dimension i.
             ==None: assume discretization is the same physical scale
                     in all directions  (i.e., deltas=1)
      
      nb: last dimension of the output is length n[-1]/2 since f(r) is real
    
    current version doesn't bother explicitly filling the f(k) array.
    Instead, makes a realization of real white noise which is then FFTed to give w(k)
    Then f(k) = sqrt[P(k)]*w(k)
    
    nb. the configuration space field is calculated with:
        numpy.fft.irfftn(dft_realizn(dims, Pk, deltas))
    
    """
    if isinstance(Pk, numbers.Number):
        if Pk<0:
            def Pkn(k):
                withnp.errstate(divide='ignore', invalid='ignore'):
                    returnnp.where(k<=0, 0, k**Pk)
        else:
            def Pkn(k):
                return k**Pk
    else:
        Pkn = Pk
    
    wr = npr.standard_normal(size=dims)  ## white noise
    wk =np.fft.rfftn(wr)
    # why no normalization here? [see getPower*() below]
    
    if Pk is None:
        return wk
    else:
        return wk*np.sqrt(Pkn(np.sqrt(get_k2(dims, deltas=deltas))))


def DFT_indices(dimensions, dtype=np.int_, dim1=None, real=False):
    """DFT_indices(dimensions,dtype=int_) returns an array representing a grid
    of DFT indices with row-only, and column-only variation.
    A DFT index is defined so that they go (0, 1, 2, .... N/2, -(N/2-1), .... -1)
    (see numpy.dft.fft)
    
    for a real DFT, the final index only runs 0...N/2
       (nb., in that case dimensions=the full dimensionality in configuration space)
    
    if dim1 is specified, only return the indices along a single dimension
       (dim1 must be iterable)
    """
    
    dims = list(dimensions)
    ndims = len(dims)
    if real: dims[-1]=dims[-1]/2+1
    tmp =np.ones(dims, dtype)
    lst = []
    
    if dim1 is None:
        dim1 = range(ndims)
    
    for i in dim1:
        d = dims[i]
        fidx =np.add.accumulate(tmp, i, )-1
        
        if not (real and i==ndims-1):
            fidx =np.where(fidx<=d/2, fidx, fidx-d)
        lst.append(fidx)
    
    returnnp.array(lst)


def get_k2(dims, deltas=1):
    """ return an array giving the value of |k|^2 for a real DFT matrix
        
        nb. dims are the full dimensionality in configuration space
        
        deltas: scale factors for converting integers k[i] into spatial
        frequencies: kphys[]=k[]/deltas[i]*scale[i]) for [] in dimension i.
        """
    
    if deltas is None:
        deltas=1
    
    freqs = 1.0/np.array(deltas)/np.array(dims)
    
    rdims = list(dims)
    rdims[-1]=rdims[-1]/2+1
    k2 =np.zeros(shape=rdims, dtype=np.float64)
    for i, freq in enumerate(freqs):
        ### get only a single dimension
        idx = DFT_indices(dims, dim1=[i], real=True)
        k2 += (idx[0]*freq)**2
    
    return k2


def test_rlzn(nr=1000, shape=(16, 256), Pk=None):
    
    ntot =np.product(shape)
    
    rftshape = list(shape)
    rftshape[-1] //= 2; rftshape[-1]+=1
    rftshape=tuple(rftshape)
    
    avg_fk =np.zeros(shape=rftshape, dtype=np.complex128)
    avg_fk2 =np.zeros(shape=rftshape, dtype=np.float64)
    
    for i in range(nr):
        fk = dft_realizn(shape, Pk=Pk)/math.sqrt(ntot)
        avg_fk += fk
        avg_fk2 += fk.real**2 + fk.imag**2
    
    return avg_fk/nr, avg_fk2/nr

