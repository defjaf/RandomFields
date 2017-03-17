"""
make a realization of an 'ill-proportioned' toroidal universe with
given P(k), excise a small cubic box, and examine the vector power
spectrum P(|k|, angle)

(Actually much more general than that.)

"""

## TODO:
#### need to make into a constant Delta f=i/Delta
#### maybe make a class that carries along delta with it?
#### split off excising/topology code (or at least make more clear in documentation
####    that this is really a generally n-dim spectrum code

from __future__ import division
from __future__ import print_function

import math
from itertools import tee

try:
    from itertools import izip
except ImportError:
    izip = zip

import numpy as N
import numpy.random as Nr
from matplotlib import pyplot as plt

import realization

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

# 
# def pairwise(iterable):
#     "s -> (s0,s1), (s1,s2), (s2, s3), ..."
#     a, b = tee(iterable)
#     try:
#         b.next()
#     except StopIteration:
#         pass
#     return izip(a, b)

 


def excise(rlzn, slices):
    """
    excise a region of dimensions n[...] from the realization rlzn.

    slices is a sequence of elements correspinding to slices
    e.g., ( 3, 3, 3 ) for a 3x3x3 submatrix starting at 0,0,0
          ( (1,4), (1,4), (1,4) ) for a  3x3x3 submatrix starting at 1,1,1
    """
    return rlzn[[slice(*s) for s in slices]]

def getAngDist(rlzn, kbin, deltas=None, nk=10, isFFT=False):
    """
    get the actual values of rlzn_k for k in a linear bin,
    as a function of angle
    
    need to normalize by the power -- but need to be more fine-grained than the bincd
    """
    # probably should have fewer angular bins for low k
    rshape = rlzn.shape
    if isFFT:
        rshape[-1] = 2*rshape[-1]-2
        rlznk = rlzn
    else:
        rlznk = N.fft.rfftn(rlzn)
        
    k = N.sqrt(realization.get_k2(rshape, deltas=deltas))
    
    idxsx = N.squeeze(realization.DFT_indices(rshape, dim1=[0], real=True))
    idxsy = N.squeeze(realization.DFT_indices(rshape, dim1=[1], real=True))
    ang = N.arctan2(idxsy, idxsx)

    kk = N.linspace(0, k.ravel().max(), nk)
    kdxi = N.logical_and(k>kk[kbin-1], k<=kk[kbin])
    ki =  (kk[kbin-1]+kk[kbin])/2.0
    print('k=',ki)

    return ang[kdxi], rlznk[kdxi], ki
    

def getPower(rlzn, deltas=None, nk=10, nangle=None, isFFT=False, eq_vol=False):
    """
    get the power spectrum of the realization, possibly in angular bins
    returns k, mean[P(k)], std[P(k)]
    
    
    """
    
    ## can use & instead of logical_and???
        
    if nangle is None or nangle < 1:
        nangle = 1
    
    if deltas is None:
        deltas=1
        
    rshape = rlzn.shape
    if isFFT:
        rshape[-1] = 2*rshape[-1]-2
        
    ndims = len(rshape)

    k = N.sqrt(realization.get_k2(rshape, deltas=deltas))

    if nangle>1:
        idxsx = N.squeeze(realization.DFT_indices(rshape, dim1=[0], real=True))
        idxsy = N.squeeze(realization.DFT_indices(rshape, dim1=[1], real=True))
        ang = N.arctan2(idxsy, idxsx)
    
    # probably should have fewer angular bins for low k
    if isFFT:
        power = N.absolute(rlzn)**2
    else:
        power = N.absolute(N.fft.rfftn(rlzn))**2
    
    if eq_vol:  ## actually want uniform in k**ndims
        kk = N.linspace(0, k.max()**ndims, nk)**(1.0/ndims)
    else:
        kk = N.linspace(0, k.max(), nk)

    kout = N.zeros_like(kk)
    aa = N.linspace(0, 2*N.pi, nangle+1)
    Pk = N.empty(shape=(nk,nangle), dtype=N.float64)
    Sk = N.empty_like(Pk)
    
    Pk[0,:] = power.flat[0]   ## 0 is always the first DFT index...
    Sk[0,:] = 0

    for ii,ki in enumerate(pairwise(kk)):
        kdxi = N.logical_and(k>ki[0], k<=ki[1])
        for jj, aj in enumerate(pairwise(aa)):
            if nangle>1:
                adxj = N.logical_and(ang>aj[0], ang<=aj[1])
                idx = N.logical_and(kdxi, adxj)
            else:
                idx = kdxi
            Pk[ii+1, jj] = power[idx].mean()
            Sk[ii+1, jj] = power[idx].std()
        kout[ii+1] = N.mean(ki)

        
    Pk = N.squeeze(Pk)
    Sk = N.squeeze(Sk)

    volume = (N.array(deltas)*N.array(rshape)).prod()
    Pk /= volume
    Sk /= volume
#     print("Volume=", volume)
    ## normalization needed for 'volume factor' 
    #      <dk dk'> = delta(k+k')P(k) => <d^2>=Vol*P(k)
    
    ## or always return the same thing?
    if nangle > 1: 
        return (kout, aa), Pk, Sk
    else:
        return kout, Pk, Sk



def driver(n=0, dims=(512,512), deltas=None, ex=2.0):

    print('dims=', dims)
    
    Pk = n
    
    delta_k = realization.dft_realizn(dims, Pk, deltas=deltas)

    print('delta_k: shape=', delta_k.shape)

    delta_r = N.fft.irfftn(delta_k)

    print('delta_r: shape=', delta_r.shape)

    assert delta_r.shape==dims

    #### plot the map
    plt.figure(0)
    plt.imshow(delta_r)
    plt.axis('scaled')
    plt.title(r'$\delta({\bf r}$)')
    
    plt.figure(4)
    plt.imshow(N.abs(delta_k))
    plt.axis('scaled')
    plt.title(r'$\delta_{\bf k}$')
    
    print('map plotted')
    

    ## plot the power spectra
    k, P, S = getPower(delta_r, nk=20)

    plt.figure(1)
    plt.loglog(k[1:], P[1:], '.')
    plt.loglog(k[1:], P[1:]+S[1:])
    plt.xlabel(r'$k$')
    plt.ylabel(r'$P(k)$')
    
    print('full spectra plotted')

    ###nb. each entry is a tuple that will be turned into a slice
    ex_dims = [(int(min(dims)/ex),)]*len(dims)
    print('Excising region of dimensions: ', ex_dims)
    ex_delta_r = excise(delta_r, ex_dims)

    ek, eP, eS = getPower(ex_delta_r, nk=20)
    plt.loglog(ek[1:], eP[1:])
    plt.loglog(ek[1:], eP[1:]+eS[1:])
    
    print('excised spectra plotted')

    #plot the power distribution
    nk = 9
    nktot = 81
    plt.figure(2)
    nrc = nk-1   ## don't plot k=0
    nr = int(math.sqrt(nrc))
    nc = int(nrc/nr)
    if nr*nc<nrc: nc += 1

    for i in xrange(1,nk):
        plt.subplot(nr, nc, i)
        ang, rlzk, k = getAngDist(ex_delta_r, i, nk=nktot)
        std = N.sqrt(((N.absolute(rlzk))**2).mean())
        plt.plot(ang, rlzk.real, '.')
        plt.plot(ang, rlzk.imag, '.')
        plt.plot([0,math.pi], [std,std], 'r')
        plt.plot([0,math.pi], [-std,-std], 'r')
        ax = plt.gca()
        lab = r'$k=%f$' % k
        plt.text(0.1, 0.9, lab, transform = ax.transAxes)

    plt.figtext(0.5, 0.93, r'$\delta(k,\theta)$')
