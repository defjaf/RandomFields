from __future__ import division
import aniso
import numpy as np
import math

def test_rlzn(nr=1000, shape=(16, 256)):
    
    ntot =np.product(shape)
    
    rftshape = list(shape)
    rftshape[-1] //= 2; rftshape[-1]+=1
    rftshape=tuple(rftshape)
    
    avg_fk =np.zeros(shape=rftshape, dtype=np.complex128)
    avg_fk2 =np.zeros(shape=rftshape, dtype=np.float64)
    
    for i in range(nr):
        fk = aniso.dft_realizn(n=shape)/math.sqrt(ntot)
        avg_fk += fk
        avg_fk2 += fk.real**2 + fk.imag**2
    
    return avg_fk/nr, avg_fk2/nr
    