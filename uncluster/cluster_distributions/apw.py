from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import numpy as np
from scipy.optimize import root

from gala.potential import HernquistPotential
from gala.units import galactic

__all__ = ['sample_radii']

# HACK: this is a BY EYE "fit" to the Sersic density profile
gc_density = HernquistPotential(m=1., c=2., units=galactic)

@u.quantity_input(r_max=u.kpc)
def sample_radii(r_max=100*u.kpc, size=1):
    """
    Use inverse transform sampling to generate samples from a Hernquist mass profile
    approximation to Oleg's Sersic profile.

    Parameters
    ----------
    r_max : `~astropy.units.Quantity` [length] (optional)
        The maximum radius or upper-bound for sampling.
    size : int, tuple (optional)
        The shape of the output array.

    Returns
    -------
    radii : `~astropy.units.Quantity` [kpc]

    """
    r_max = r_max.to(u.kpc).value
    Menc = lambda r: gc_density.c_instance.mass_enclosed(np.array([[r,0,0]]),
                                                         G=gc_density.G)[0]

    def root_func(r, m):
        return (m - Menc(float(r)))

    if r_max == np.inf:
        m_max = 1.
    else:
        m_max = Menc(r_max)

    m = np.random.uniform(0., m_max, size=size)
    return np.array([root(root_func, 1., args=(m[i],)).x[0] for i in range(size)]) * u.kpc
