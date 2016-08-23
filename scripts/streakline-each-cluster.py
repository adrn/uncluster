"""
    Given the masses, mass-loss rates, orbital radii, and destruction times for an initial
    population of clusters computed from reproducing the simulation in Gnedin et al. (2014),
    generate Streakline+scatter+full-disruption simulations for each stream.

    This code makes the following simplifying assumptions:
        - Dynamical friction is neglected: we are interested in clusters with r > 5 kpc, where,
          even for the most massive globular clusters, dynamical friction has a negligible effect.
        -

    TODO:
    - Need to do a grid over eccentricity distribution parameters OR anisotropy β for an isotropic
      DF
    - In next script, take output from this file and paint stellar population, abundances on to
      star particles in each "stream"

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

_fac = (1*u.kpc/u.Myr).to(u.km/u.s).value
def _circ_vel_kms(r, pot):
    r = float(r)
    dPhi_dr = pot._gradient(np.array([r,0.,0.]))[0]
    return np.sqrt(np.abs(r*dPhi_dr)) * _fac

def P(r, pot):
    return 41.4 * r / _circ_vel_kms(r, pot)

def t_tid(r, M, pot, α=2/3.):
    """ Tidal disruption timescale """
    return 10. * (M / 2E5)**α * P(r, pot)

def t_iso(M):
    """ Isolation disruption timescale (2-body evaporation)"""
    return 17. * (M / 2E5)

def t_df(r, M, pot, f_e=0.5):
    return 0.45 * r*r * _circ_vel_kms(r, pot) * (M/1E5)**-1 * f_e

def F(y, t):
    M,r2 = y

    r = np.sqrt(r2)
    min_t = np.min([t_tid(r, M, pot),
                    t_iso(M)], axis=0)

    M_dot = -M/min_t
    r2_dot = -r2/t_df(r, M, pot)

    return [float(M_dot), float(r2_dot)]

class GlobularCluster(object):

    @u.quantity_input(r=u.kpc, mass_initial=u.Msun)
    def __init__(self, radius_initial, mass_initial, ecc):
        self.r_i = radius_initial
        self.m_i = mass_initial
        self.ecc = float(ecc)

    @u.quantity_input(t_grid=u.Myr)
    def mass_history(self, t_grid):
        """
        Compute the mass evolution of the cluster by integrating

            dM/dt = M / min(t_iso, t_tid)

        Parameters
        ----------
        t_grid : quantity_like
            An array of times to compute the mass of the cluster at.
        """

        _M = self.m_i.decompose(galactic).value
        _r = self.r_i.decompose(galactic).value
        _t_grid = t_grid.decompose(galactic).value

        M_r2 = odeint(F, [_M, _r**2], t=_t_grid)

def main():

    # potential used in Gnedin et al. 2014
    pot = gp.CCompositePotential()
    pot['stars'] = bscf.SCFPotential(m=sersic_dens.m, r_s=sersic_dens.Re,
                                     Snlm=S, Tnlm=np.zeros_like(S), units=galactic)

    v_c = np.sqrt(G * 1E12*u.Msun / (20*u.kpc) * (np.log(2.)-0.5))
    v_c = galactic.decompose(v_c)
    print(v_c)
    pot['halo'] = gp.SphericalNFWPotential(v_c=v_c, r_s=20*u.kpc, units=galactic) # M(<250kpc) = 10^12 Msun

    pot.mass_enclosed([250.,0,0]*u.kpc) # mass in ~virial radius


if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")

    parser.add_argument("-f", dest="field_id", default=None, required=True,
                        type=int, help="Field ID")
    parser.add_argument("-p", dest="plot", action="store_true", default=False,
                        help="Plot or not")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main()
