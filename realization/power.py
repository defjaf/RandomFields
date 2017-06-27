#!/usr/bin/env python
# encoding: utf-8
"""
power.py

cosmological power spectra

Created by Andrew H. Jaffe on 2011-01-25.
Copyright (c) 2011 Imperial College London. All rights reserved.
"""

import math
import numpy as np
import scipy.integrate as spi

class PowSpec(object):
    """class for calculating cosmological power spectra (and keeping track of normalizations)"""

    def __init__(self, rawPk, smooth=0):
        self.rawPk = rawPk
        self.norm = None
        self._params = None
        self._sigma8 = None
        self.smooth=smooth


    def __call__(self, k, sigma8=1, recalc=False, **kwargs):

        def Pk(kk):
            return self.rawPk(kk, **kwargs)

        if self.smooth > 0:   ## well, don't want this for the sigma8 calc...
            def Pksmooth(kk):
                return np.exp(-(kk*self.smooth)**2)*self.rawPk(kk, **kwargs)
        else:
            Pksmooth = Pk


        if (recalc or self.norm is None
           or self._sigma8 != sigma8 or self._params!=kwargs):
            sig2_8 = sigma2_R(Pk, 8)
            self.norm = sigma8**2/sig2_8
            self._sigma8 = sigma8
            self._params = kwargs
            # print "got new normalization=", self.norm

        return self.norm*Pksmooth(k)


def bbks(k, ns=1, Omegam=0.3, h=0.72):
    """return unnormalized bbks CDM power spectrum with tilt n_s"""

    with np.errstate(invalid='ignore'):  # catch k==0
        q = k/(Omegam*h)  ## only one power of h so [k] = h/Mpc
        T2cdm = (np.log(1+2.34*q)/(2.34*q))**2
        T2cdm /= np.sqrt(1+3.89*q+(16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)

    return np.where(k==0, 0, k**ns * T2cdm)

def gaussian(k, sigma_k=1.0):
    """ return unnormalized Gaussian power spectrum exp[-(k/sigma_k)^2/2] """
    return np.exp(-0.5*(k/sigma_k)**2)


def sigma2_R(Pk, R=8, kpow=0, prefac=1):
    """
    calculate the amplitude sigma_R^2 of power spectrum Pk(k).
    if kpow!=0, calculate the variance of the quantity
       with spectrum k**kpow * prefac * Pk(k)
       (e.g., for <v^2>_R, kpow=-2, prefac=(a_0 H_0 f_grav)**2
    """

    def PkIntegrand(k):
        """ \integrate k^3 P(k)/2pi^2  W^2(kR) dk/k"""
        kR = k*R
        Wk = 3*(np.sinc(kR/math.pi)-np.cos(kR))/kR/kR  # np.sinc=sin(pi x)/(pi x)  !!
        return k**(2+kpow) * Wk*Wk * Pk(k)/(2*math.pi*math.pi)

    # def PkIntegrand_lnk(lnk):
    #     return PkIntegrand(np.exp(lnk))*np.exp(lnk)

    s2, err =  spi.quad(PkIntegrand, 0.0, np.Inf)
#    s2, err =  spi.quad(PkIntegrand_lnk, -30, 30)

    if err/s2 > 0.1:
        print "sig2 calc: err, s2 = ", err, s2

    return s2*prefac



powspec = PowSpec(bbks)
