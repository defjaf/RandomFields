#### UNUSED
def getPk(*idxs, **kwargs):    ##dims=None, scale=None):
    print 'getPk: idxs=', idxs
    print '       kwargs=', kwargs
    d = N.array(kwargs.dims)
    for i in zip(*idxs):
        kdxs = N.where(i<=d/2, i,  i-d)
        if kwargs.scale is not None:
            kdxs/=N.array(kwargs.scale)
        sum += kdxs**2.
    return (kdxs**2).sum()

#### UNUSED -- see numpy.dft.fftfreq(n)
def frange(n):
    """ return the integer frequencies associated with a DFT of length n 
        (only works for n even!)
    """
    f = N.arange(n, dtype=N.int32)
    f[n/2+1:] = N.arange(-n/2+1,0, dtype=N.int32)
    return f
    


def dft_realizn_bad(n, Pk, scale=1):
    """
    THIS CODE IS WRONG (wrong real and complex-conjugates in the array)
    """
    if not N.all(n == 2*(N.array(n)//2)):
        raise NotImplementedError, ("Need even number of elements in each dimension: ",n)
    
    nn = N.array(n)
    nn[-1] = nn[-1]/2+1   ## real DFT is smaller than n*n -- only +ve freqs in one dimension
    
    ## fill with appropriate complex numbers -- THIS IS WRONG
    ##     rlzn[..., 1:-1] are complex with unit variance, so <Re^2>=<Im^2>=1/2
    ##     rlzn[...,0] and [...,-1] are real 

    ntot = N.prod(nn)
    nreal = N.prod(nn[:-1])
    ncomp = ntot-2*nreal
    
    rlzn = N.empty(shape=nn, dtype=N.complex128)

    rlzn[..., 1:-1] = (Nr.normal(size=ncomp) + Nr.normal(size=ncomp)*1.0j)/2
    rlzn[..., 0] = Nr.normal(size=nreal) 
    rlzn[..., -1] = Nr.normal(size=nreal)

    return rlzn*N.sqrt(Pk(N.sqrt(get_k2(n, scale=scale))))    
