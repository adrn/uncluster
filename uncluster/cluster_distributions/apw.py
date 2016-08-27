from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import numpy as np
from scipy.optimize import root

from gala.potential import HernquistPotential
from gala.units import galactic

# use the same mass function as Gnedin
from .gnedin import sample_masses

__all__ = ['gc_prob_density', 'sample_radii', 'sample_masses']

# HACK: this is a BY EYE "fit" to the Sersic density profile
_hernquist = HernquistPotential(m=1., c=2., units=galactic)
def gc_prob_density(r):
    """
    Evaluate the **probability** density of the spatial distribtuon
    of globular clusters following a Hernquist profile.

    This is *not* the mass-density or number-density, but:

    .. math::

        \nu(r) = \int f(r,v)\,{\rm d}v

    Parameters
    ----------
    r : float
        Radius in kpc.

    """
    return _hernquist.c_instance.density(np.array([[r,0,0]]))[0]

def sample_radii(size=1):
    """
    Use inverse transform sampling to generate samples from a Hernquist mass profile
    approximation to Oleg's Sersic profile.

    Parameters
    ----------
    size : int, tuple (optional)
        The shape of the output array.

    Returns
    -------
    radii : `~astropy.units.Quantity` [kpc]

    """
    Menc = lambda r: _hernquist.c_instance.mass_enclosed(np.array([[r,0,0]]),
                                                         G=_hernquist.G)[0]

    def root_func(r, m):
        return (m - Menc(float(r)))

    m = np.random.uniform(0., 1., size=size)
    return np.array([root(root_func, 1., args=(m[i],)).x[0] for i in range(size)]) * u.kpc
