from __future__ import division, print_function

# Standard library
import time

# Third-party
from astropy import log as logger
import numpy as np

from ..core import solve_mass_evolution
from ...potential import mw_potential

def test():

    t_grid = np.linspace(-11., 0., 4096)
    m0 = 503091.704715
    r_grid = np.full_like(t_grid, 18.6115039789)
    i,Ms = solve_mass_evolution(m0, t_grid, r_grid)

    print(i)
    print(Ms)
