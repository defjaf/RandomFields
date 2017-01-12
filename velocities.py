#!/usr/bin/env python
# encoding: utf-8
"""
velocities.py

make a realization of a n-d gaussian random field, 
derive its velocities from poisson+gravity, 
and make an n-body realization with local number density proportional to matter density

March 2016: split into vanilla density field and field with physical velocities
   (but see realization.py which has all the actual DFT stuff for a density field)
   
TODO: add something that explcitly gets the k values for the fourier-space version?
    [using rlzn.get_k(), rlzn.get_k2()]
    nb. get_k* give frequency, NOT angular frequency

Created by Andrew H. Jaffe on 2009-12-11.
Copyright (c) 2009 Imperial College London. All rights reserved.
"""

from __future__ import division
from __future__ import with_statement

import math
import sys
import os
import time

from optparse import OptionParser

import numpy as np

canPlot = True

try:
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
except ImportError:
    canPlot = False
    print "can't plot"

import numpy.random as rnd

import realization as rlzn

import power

### NB fourier conventions
## rfftn: F_k = \sum_n f_n exp(-2pi ikn/N)
## irfft: f_n = (1/N) \sum_k F_k exp(2pi ikn/N)

Omega_m = 0.3
hubble = 0.72

a0H0 = 100*hubble ### km/s/Mpc
fgrav = Omega_m**0.6

class Field(object):
    """
    class holding information for a n-d density field
    I think that nothing in the class requires n=3.
    
    """
    def __init__(self, delta_k=None, delta_r=None, deltas=1):
        if delta_k is None and delta_r is None :
            raise Exception("must set one of delta in real or fourier")
        
        self._delta_k = delta_k
        self._delta_r = delta_r
                
        if delta_k is not None:
            self.ndim = len(delta_k.shape)
        elif delta_r is not None:
            self.ndim = len(delta_r.shape)
        
        self.deltas = deltas
        
                                
    @property
    def delta_k(self):
        "density in fourier space"
        if self._delta_k is None:
            if self._delta_r is not None:
                self._delta_k = np.fft.rfftn(self._delta_r)
            else:
                raise Exception("must have delta_r set for delta_k access")
            
        return self._delta_k
            
            
    @delta_k.setter
    def delta_k(self, value):
        self._delta_k = value
        
    @delta_k.deleter
    def delta_k(self):
        self._delta_k = None
        
    @property
    def delta_r(self):
        "density in real space"
        if self._delta_r is None:
            if self._delta_k is not None:
                self._delta_r = np.fft.irfftn(self._delta_k)
            else:
                raise Exception("must have delta_k set for delta_r access")
        return self._delta_r

    @delta_r.setter
    def delta_r(self, value):
        self._delta_r = value

    @delta_r.deleter
    def delta_r(self):
        self._delta_r = None
        
        
class FieldWithVelocities(Field):
    """class holding information for a n-d density and velocity field
    I think that nothing in the class requires n=3 
    (but the catalog() method makes most sense for n=3)
    
    """
    
    def __init__(self, delta_k=None, delta_r=None, v_k=None, v_r=None, deltas=1,
                 dipole_rms=0):
        if delta_k is None and delta_r is None and v_k is None and v_r is None:
            raise Exception("must set one of v or delta in real or fourier")
            
        Field.__init__(self, delta_k=delta_k, delta_r=delta_r, deltas=deltas)

        self._v_k = v_k
        self._v_r = v_r
                
        self.fgrav = 1 ### velocity "growth factor"
        
        self.dipole_rms = dipole_rms 

        
    @property
    def v_k(self):
        "velocity in Fourier space (vector)"
        if self._v_k is None:
            
            ### v_k = +i a H f  delta_k {\vec k}/k^2 = +iaHf d_k/|k| khat; f=dlnD/dlna
            #### nb '+' above -- due to numpy discrete FFT convention opposite to cosmology
            ### deltas give conversion to physical (non angular) frequency for k/k^2
            k2 = rlzn.get_k2(self.delta_r.shape, deltas=self.deltas)    #shape rdims
            kvec = rlzn.get_k(self.delta_r.shape, deltas=self.deltas)   #shape (ndim, *rdims)
            
            with np.errstate(invalid='ignore'):  # catch k==0
                self._v_k = 1j*np.array([self.delta_k*kveci/k2 for kveci in kvec])  
            self._v_k *= a0H0*fgrav/(2*math.pi)   ## 2pi converts k/k^2 to angular frequency
            
            for i in range(self.ndim):   
                dipole_i = np.random.standard_normal()*self.dipole_rms/math.sqrt(self.ndim)
                self._v_k[i,:].flat[0] = dipole_i ## take care of k=0 
            
        return self._v_k

    @v_k.setter
    def v_k(self, value):
        self._v_k = value

    @v_k.deleter
    def v_k(self):
        self._v_k = None

    @property
    def v_r(self):
        "velocity in real space"
        if self._v_r is  None:
                self._v_r = np.array([np.fft.irfftn(vk) for vk in self.v_k])
            
        return self._v_r

    @v_r.setter
    def v_r(self, value):
        self._v_r = value

    @v_r.deleter
    def v_r(self):
        self._v_r = None

    def catalog(self, nbar, verbose=False, lowmem=False):
        """
        make a Poisson-process mock catalog from this field. 
        The (poisson) mean number of points in the entire realization is nbar, 
        so the mean in pixel p is nbar*(1+delta_p)/N
           where N is the total number of pixels 
           [since N ~ rho_tot = \sum_p (1+delta_p) = N + \sum_p delta_p = N*(1+delta_p/N_p) ]
        """
        if lowmem and verbose: 
            sys.stderr.write('%s: creating catalog\n' % time.asctime())

        nmeans = 1+self.delta_r
        nmeans[nmeans<0]=0.0
        ## adjust so that it has the correct mean *after* clipping rho<0
        nmeans *= nbar/np.size(self.delta_r)/nmeans.mean()  

        if lowmem: 
            del self.delta_r
            if verbose: 
                sys.stderr.write('%s: deleted delta_r\n' % time.asctime())

        npoints = rnd.poisson(nmeans)
        
        del nmeans
                
        ### now, convert to locations
        if verbose:
            justright = np.sum(npoints==1)
            toomany = np.sum(npoints>1)
            sys.stderr.write('number with n==1: %d\n' % justright)
            sys.stderr.write('number with n>1: %d\n' % toomany)
        
        idxs = np.transpose(np.nonzero(npoints>0))
        
        if lowmem:
            self.v_r   ## so it gets accessed and therefore created.
            del self.v_k
            if verbose:
                sys.stderr.write('%s: got v_r, deleted v_k\n' % time.asctime())
            

        ### extract the n-d velocities
        ### this can definitely be done in a single line...
        if self.ndim == 3: 
            vels = np.array([self.v_r[:, idxs[i][0], idxs[i][1], idxs[i][2]] 
                             for i in range(len(idxs))])
        elif self.ndim == 2:
            vels = np.array([self.v_r[:, idxs[i][0], idxs[i][1]] for i in range(len(idxs))])
        elif seld.ndim == 1:
            vels = np.array([self.v_r[:, idxs[i][0]] for i in range(len(idxs))])
        else:
#            print("currently only works with ndim=1,2,3..."
            vels = np.array([])
        
        if lowmem:
            del self.v_r
            if verbose:
                sys.stderr.write('%s: deleted v_r\n' % time.asctime())
            
        return self.deltas*idxs, vels


def processCatalog(posns, vels, verr=0, nonlin=0, ctr=(0,0,0), lens=(1,1,1)):
    """
       re-center catalog at an arbitrary point in the volume
       convert n-d velocity catalog to radial velocities. 
       add nonlinearity correction and noise if needed.
       TODO: possibly use zeldovich approx to displace points?
             (optionally) adjust the radial coordinate to take into account the peculiar velocity?
    """
    
    lens = np.array(lens)
    ctr = np.array(ctr)
    
    ndims = len(ctr)

    ## center positions
    pos = np.mod(posns-ctr, lens)-lens/2.0
    
    nl_vels = vels.copy()
    if nonlin > 0:
        nl_vels += np.random.standard_normal(vels.shape)*nonlin/math.sqrt(ndims)
        
    vr = (pos*nl_vels).sum(axis=-1)/np.sqrt((pos**2).sum(axis=-1))
    
    if verr > 0:
        vr += np.random.standard_normal(vr.shape)*verr
        
    return pos, vr
    

    
def plotcatalog(cat):
    """3d plot of a catalog of points """
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cat[:,0], cat[:,1], cat[:,2])
    plt.show()
    
    
def toPolar(pos, deg=True):
    """ convert a shape=(:, 3) cartesian coordinate array to polars """
    r = np.sqrt((pos**2).sum(axis=-1))
    if deg:
        theta = 90-np.degrees(np.arccos(pos[:,2]/r))
        phi = np.degrees(np.arctan2(pos[:,1], pos[:,0]))
    else:
        theta = np.pi/2-np.arccos(pos[:,2]/r)
        phi = np.arctan2(pos[:,1], pos[:,0])
    return np.vstack((r,theta,phi)).transpose()
    
    
def generate_velocity_catalog(N=256, nbar=500, length=100, verr=1000, nonlin=250,
                    plot=False, stdout=False, ret=True, ndim=3):
    """
       generate mock position and velocity catalog.
       options are as in main program.
       use RMS in sphere with R=length/2 for v(k=0)
       
       works for ndim!=3 but not clear that it means anything...
   
    """
       
    shape = (N,)*ndim
    deltas = length/np.asarray(shape)
       
    ntot=np.product(shape)
    rftshape = list(shape)
    rftshape[-1] //= 2; rftshape[-1]+=1
    rftshape=tuple(rftshape)
    
    powspec = power.PowSpec(power.bbks, smooth=0)
    
    fk = rlzn.dft_realizn(shape,Pk=powspec, deltas=deltas) 
    
    dipole2 = power.sigma2_R(powspec, R=length/2.0, kpow=-2, prefac=(a0H0*fgrav)**2)
    dipole_rms = math.sqrt(dipole2)     ## REALLY SHOULD BE FULL BOX, NOT SPHERE
    print "#dipole v_rms = %f" % dipole_rms

    field = FieldWithVelocities(delta_k=fk, deltas=deltas, dipole_rms=dipole_rms)
    
    pos, vel = field.catalog(nbar, verbose=True, lowmem=False)
    
    if not len(pos):
        raise Exception("Uh oh, no points")
    
    ctr = (0,)*ndim
    pos, vr = processCatalog(pos, vel, lens=length, verr=verr, nonlin=nonlin, ctr=ctr)

    if plot:
        plotcatalog(pos)
    
    if ndim==3:
        polar = toPolar(pos)
    
        if stdout:
            print ("#%5s %8s %8s %8s %8s %8s %8s %8s" %
                   ("i", "r", "theta", "phi" ,"v_x","v_y","v_z", "S_r"))
            for i, xyz_vxyz_vr in enumerate(zip(polar, vel, vr)):
                (r,t,p), (vx, vy, vz), vr  = xyz_vxyz_vr
            
                print (" %5d %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f" %
                       (i, r,t,p, vx,vy,vz, vr))
    else:
        print("No output for ndim=%d" % ndim)

    if ret:
        return field, pos, vel, vr
    
    
test = generate_velocity_catalog

class ahjoptions:
    pass

def driver():
    """ do a specific set of runs  """
    
    nrun = 6
    ngal = 4000
    Lbox = 200 
    verr = 1660
    nlstd = 250
    
    opts = ahjoptions()
    opts.npoints = 256
    opts.nbar = ngal
    opts.length = Lbox
    opts.verr = verr
    opts.nonlin = nlstd
    opts.plot = False
    
    
    for i in range(nrun):
        fname = "SFI_mock_%d.txt" % i
        print "file ", fname
        old_stdout = sys.stdout
        with open(fname, 'w') as sys.stdout:
            main(opts)
        sys.stdout = old_stdout



def main(options=None):
    
    if options is None:
        parser = OptionParser()
        parser.add_option("-n", "--npoints", type="int", default=256,
                          help="number of points in each direction")
        parser.add_option("--nbar", type="int", default=500,
                          help="mean number of samples")
        parser.add_option("--length", "-l", type="float", default=100,
                          help="box size (Mpc/h)")
        parser.add_option("--verr", type="float", default=1000,
                          help="1-d velocity error std (km/s)")
        parser.add_option("--nonlin", type="float", default=250,
                          help="3-d nonlinearity std (km/s)")
        parser.add_option("--plot", "-p", action="store_true", default=False,
                          help="plot with matplotlib")
        (options, args) = parser.parse_args()
    else:
        print ("# npoints=%d nbar=%d length=%6.1fMpc verr=%6.1fkm/s nonlin=%6.1fkm/s" %
               (options.npoints, options.nbar, options.length, options.verr, options.nonlin))

    generate_velocity_catalog(N=options.npoints, nbar=options.nbar, length=options.length, 
                     verr=options.verr, nonlin=options.nonlin,
                     plot=options.plot, stdout=True, ret=False)
         

if __name__ == '__main__':
    main()

