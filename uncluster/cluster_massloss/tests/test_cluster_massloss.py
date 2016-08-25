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
    m0 = 53091.704715
    r0 = 18.6115039789

    solve_mass_radius(m0, r0, t_grid)
