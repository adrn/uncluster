from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import numpy as np
from scipy.special import gammainc
from scipy.interpolate import InterpolatedUnivariateSpline

__all__ = ['sample_masses', 'sample_radii']

@u.quantity_input(M_min=u.Msun, M_max=u.Msun)
def sample_masses(M_min, M_max, size=1):
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

def _sersic_frac_mass_enclosed_grid(rmax, n_s=2.2, R_e=4., n_grid=1024):
    """
    This was adapted from Oleg's C code (uncluster/src/gc_evolution.c).
    For the Milky Way, the parameters (r_s, n_s, etc.) don't evolve and z=0.
    The default value for n_grid is a MAGIC NUMBER
    """
    an = 2 * n_s
    bn = 2. * n_s - 1./3. + 0.0098765/n_s + 0.0018/n_s**2 # approximation from Ciotti ...something

    argc = bn * (rmax/R_e) ** (1./n_s)
    gamnorm = gammainc(an, argc)

    r_grid = np.logspace(-4, np.log10(rmax), n_grid) # kpc
    arg = bn * (r_grid/R_e)**(1./n_s)
    m_grid = gammainc(an, arg) / gamnorm

    return r_grid, m_grid

@u.quantity_input(r_max=u.kpc)
def sample_radii(r_max, size=1):
    """
    Use inverse transform sampling to generate samples from a Sersic
    mass profile (following Gnedin et al. 2014).

    Parameters
    ----------
    r_max : `~astropy.units.Quantity` [length]
        The maximum radius or upper-bound for sampling.
    size : int, tuple (optional)
        The shape of the output array.

    Returns
    -------
    radii : `~astropy.units.Quantity` [kpc]

    """
    r_grid, m_grid = _sersic_frac_mass_enclosed_grid(rmax=r_max.to(u.kpc).value)
    interp_func = InterpolatedUnivariateSpline(m_grid, r_grid, k=1)
    return interp_func(np.random.uniform(0, 1, size=size)) * u.kpc
