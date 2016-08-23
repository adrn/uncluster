"""
    Generate masses and mean orbital radii for a sample of globular clusters following
    Gnedin et al. (2014).
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import h5py
import numpy as np
from scipy.integrate import odeint

import gala.coordinates as gc
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
from gala.units import galactic

import biff.scf as bscf

def sample_masses(M_min, M_max, size=1):
    M_min = M_min.to(u.Msun).value
    M_max = M_max.to(u.Msun).value

    class MassFunction(rv_continuous):
        def _pdf(self, x):
            return x**-2 / (1/self.a - 1/self.b)

    return MassFunction(a=M_min, b=M_max).rvs(size=size) * u.Msun

def sample_radii(pot, r_min=0.*u.kpc, r_max=np.inf*u.kpc, size=1):
    r_min = r_min.to(u.kpc).value
    r_max = r_max.to(u.kpc).value

    # Menc = lambda rr: quad(lambda r: 4*np.pi*r**2*pot._density(np.array([r,0,0])), 0, rr)[0]
    Mtot = pot.parameters['m'].value
    Menc = lambda rr: pot.mass_enclosed([rr,0.,0.]).value

    def root_func(r, m):
        return (m - Menc(float(r))/Mtot)[0]

    if r_min == 0.:
        m_min = 0.
    else:
        m_min = Menc(r_min)/Mtot

    if r_max == np.inf:
        m_max = 1.
    else:
        m_max = Menc(r_max)/Mtot

    m = np.random.uniform(m_min, m_max, size=size)
    return np.array([root(root_func, 10., args=(m[i],)).x[0] for i in range(size)]) * u.kpc

