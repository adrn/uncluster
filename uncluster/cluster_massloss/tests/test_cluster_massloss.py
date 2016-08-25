from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import time

# Third-party
from astropy import log as logger
import astropy.units as u
import numpy as np

from .._core import solve_mass_radius

def test():

    t_grid = np.linspace(0, 11.5, 4096)
    m0 = np.zeros(16) + 1E6
    r0 = np.zeros(16) + 50.

    t1 = time.time()
    solve_mass_radius(m0, r0, t_grid)
    print(time.time()-t1)
