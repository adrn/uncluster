# Third-party
import astropy.units as u
import gala.potential as gp
from gala.units import galactic

# Project
from .components import (GrowingHernquistPotential,
                         GrowingMiyamotoNagaiPotential,
                         GrowingSphericalNFWPotential)
from .potential_config import m_h, r_s, m_n, c_n

# best-fit values come from running TODO:
mw_potential = gp.CCompositePotential()
mw_potential['bulge'] = GrowingHernquistPotential(m0=5E9, c0=1., units=galactic)
mw_potential['disk'] = GrowingMiyamotoNagaiPotential(m0=6.8E10*u.Msun, a0=3*u.kpc, b=280*u.pc,
                                                     units=galactic)
mw_potential['nucl'] = GrowingHernquistPotential(m0=m_n, c0=c_n, units=galactic) # best-fit values
mw_potential['halo'] = GrowingSphericalNFWPotential(m0=m_h, r_s=r_s, units=galactic) # best-fit values
