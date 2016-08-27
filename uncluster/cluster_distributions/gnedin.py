from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import numpy as np
from scipy.special import gammainc
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.misc import derivative

from ..config import M_min, M_max

__all__ = ['gc_prob_density', 'sample_masses', 'sample_radii']

def _sersic_frac_mass_enclosed(r, n_s=2.2, R_e=4.):
    """
    This was adapted from Oleg's C code (uncluster/src/gc_evolution.c).
    For the Milky Way, the parameters (r_s, n_s, etc.) don't evolve and z=0.
    """
    a_n = 2 * n_s
    b_n = 2. * n_s - 1./3. + 0.0098765/n_s + 0.0018/n_s**2 # approximation from Ciotti ...something

    # rmax = r_max.to(u.kpc).value
    # argc = b_n * (rmax/R_e) ** (1./n_s)
    # gamnorm = gammainc(a_n, argc)
    gamnorm = 1. # r_max = infinity

    arg = b_n * (r/R_e)**(1./n_s)
    return gammainc(a_n, arg) / gamnorm

# HACK: I need a function to evaluate the density profile, so I do this numerically...
# - this does a calculation on import (BAD) but it's very fast (~10s of ms)
n_grid = 4096 # MAGIC NUMBER
r_grid = np.logspace(-4, 2.5, n_grid) # kpc
m_grid = _sersic_frac_mass_enclosed(r_grid)

# if necessary, remove duplicate m's where it saturates
_idx, = np.where(m_grid >= 1.)
if len(_idx) > 1:
    r_grid = r_grid[:_idx[0]+1]
    m_grid = m_grid[:_idx[0]+1]

dm_dr = np.zeros(n_grid)
for i,r in enumerate(r_grid):
    d = derivative(_sersic_frac_mass_enclosed, r, dx=1E-3*r)
    dm_dr[i] = d

_idx = np.isfinite(dm_dr) & (dm_dr>0.)
_interp_ln_dens = InterpolatedUnivariateSpline(r_grid[_idx],
                                               np.log(dm_dr[_idx]) - np.log(4*np.pi*r_grid[_idx]**2),
                                               k=1)
def gc_prob_density(r):
    """
    Evaluate the **probability** density of the spatial distribtuon
    of globular clusters following a Hernquist profile.

    This is *not* the mass-density or number-density, but:

    .. math::

        \nu(r) = \int f(r,v)\,{\rm d}v

    .. note::

        This function computes the density numerically using linear interpolation.

    Parameters
    ----------
    r : float
        Radius in kpc.

    """
    return np.exp(_interp_ln_dens(r))

@u.quantity_input(M_min=u.Msun, M_max=u.Msun)
def sample_masses(M_min=M_min, M_max=M_max, size=1):
    """
    Use inverse transform sampling to generate samples from a power-law
    initial mass distribution with beta = -2:

        p(m) = A m^{-2}

    Parameters
    ----------
    M_min : `~astropy.units.Quantity` [mass]
        The minimum mass or lower-bound for sampling.
    M_max : `~astropy.units.Quantity` [mass]
        The maximum mass or upper-bound for sampling.
    size : int, tuple (optional)
        The shape of the output array.

    Returns
    -------
    masses : `~astropy.units.Quantity` [Msun]
        Masses sampled from the mass function.

    """

    A = 1 / (1/M_min - 1/M_max)
    R = np.random.uniform(size=size)

    return 1 / (1/M_min - R/A)

def sample_radii(size=1):
    """
    Use inverse transform sampling to generate samples from a Sersic
    mass profile (following Gnedin et al. 2014).

    Parameters
    ----------
    size : int, tuple (optional)
        The shape of the output array.

    Returns
    -------
    radii : `~astropy.units.Quantity` [kpc]

    """
    interp_func = InterpolatedUnivariateSpline(m_grid, np.log(r_grid), k=1)
    return np.exp(interp_func(np.random.uniform(0, 1, size=size))) * u.kpc
