from astropy.constants import G
import gala.potential as gp
from gala.units import galactic
import numpy as np

from .conf import M_h, rs_h

potential = gp.CompositePotential()
potential['disk'] = gp.MiyamotoNagaiPotential(m=4E10, a=3.5, b=0.14, units=galactic)
potential['bulge'] = gp.HernquistPotential(m=1E10, c=1.1, units=galactic)

v_c2 = (np.log(2.) - 0.5) * (G * M_h / rs_h)
potential['halo'] = gp.SphericalNFWPotential(v_c=np.sqrt(v_c2), r_s=rs_h, units=galactic)
