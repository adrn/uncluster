"""
    Solve for the mass-loss rates and destruction times for each cluster given
    its eccentricity, initial mass, and initial radius.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import math
from os.path import join, exists

# Third-party
from astropy import log as logger
from astropy.constants import G
from astropy.table import QTable
import astropy.units as u
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import ode, odeint

from uncluster import get_output_path, sample_radii, sample_masses
from uncluster.gnedin_mass_radius import _sersic_frac_mass_enclosed_grid
from uncluster.conf import M_tot, r_max, M_h, rs_h, t_evolve
OUTPUT_PATH = get_output_path(__file__)

# ----------------------------------------------------------------------------
# These look a bit messy and don't have units support because they
# need to be fast.
_M_tot = M_tot.to(u.Msun).value
_Mh = M_h.to(u.Msun).value
_rs_h = rs_h.to(u.kpc).value

r_grid, m_grid = _sersic_frac_mass_enclosed_grid(rmax=r_max.to(u.kpc).value)
def mass_enc_stars(r):
    if r < 1E-4:
        return 1.
    interp_func = InterpolatedUnivariateSpline(r_grid, m_grid * _M_tot, k=1)
    return interp_func(r)

kms = np.sqrt(G).to(u.km/u.s / np.sqrt(1*u.Msun/u.kpc).unit).value
def _circ_vel_kms(r):
    Mstar = mass_enc_stars(r)
    Mhalo = _Mh*(math.log(1.+r/_rs_h) - r/_rs_h/(1.+r/_rs_h))
    return math.sqrt((Mstar + Mhalo)/r) * kms # conversion to km/s

# Functions used to solve the differential equation for cluster mass M(t)
def P(r):
    return 207/5. * r / _circ_vel_kms(r) # Eq. 4

def t_tid(r, M, α=2/3.):
    """ Tidal disruption timescale """
    return 10. * (M / 2E5)**α * P(r) # Gyr, Eq. 4

def t_iso(M):
    """ Isolation disruption timescale (2-body evaporation)"""
    return 17. * (M / 2E5) # Gyr, Eq. 5

def t_df(r, M, f_e=0.5): # HACK: this f_e is WRONG -- should be computed from orbit...
    return 64. * r*r * _circ_vel_kms(r)/283. * (2E5/M) * f_e # Gyr, Eq. 8

def F(y, t):
    M,r2 = y

    if M <= 0:
        return np.array([np.nan, np.nan])

    r = math.sqrt(r2)
    min_t = min(t_tid(r, M), t_iso(M))

    M_dot = -M / min_t
    r2_dot = -r2 / t_df(r, M)

    if M_dot > 0:
        return np.array([np.nan, np.nan])

    return np.array([M_dot, r2_dot])

# TODO: could rewrite the above with Cython and implement forward Euler to make this super fast

def main():

    filename = join(OUTPUT_PATH, "1-gc-properties.ecsv")
    if not exists(filename):
        raise IOError("File '{}' does not exist -- have you run 1-make-cluster-props.py?"
                      .format(filename))
    gc_props = QTable.read(filename, format='ascii.ecsv')

    t_grid = np.linspace(0., t_evolve.to(u.Gyr).value, 4096)
    dt = t_grid[1]-t_grid[0]

    t_disrupt = np.zeros(len(gc_props))
    for j,row in enumerate(gc_props):
        # TODO: eccentricity
        _r = row['radius'].to(u.kpc).value
        _M = row['mass'].to(u.Msun).value

        y = np.zeros((t_grid.size, 2))
        y[0] = [_M, _r**2]

        # M_r2_odeint = odeint(F, y[0], t=t_grid) # TODO: why doesn't this work??

        # use a forward Euler method instead...
        for i,t in enumerate(t_grid[:-1]):
            dy_dt = F(y[i], t)
            y[i+1] = y[i] + dy_dt*dt

            if np.isnan(dy_dt[0]):
                break

        M_t = y[:i+1, 0]
        r_t = np.sqrt(y[:i+1, 1])
        t = t_grid[:i+1]

        t_disrupt[j] = t[-1]

if __name__ == '__main__':
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")

    parser.add_argument("-s", "--seed", dest="seed", default=8675309,
                        type=int, help="Random number generator seed.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    np.random.seed(args.seed)
    main()
