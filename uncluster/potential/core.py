# Third-party
import astropy.units as u
import gala.potential as gp
from gala.units import galactic

# Project
from .mwpotential import (GrowingHernquistPotential,
                          GrowingMiyamotoNagaiPotential,
                          GrowingSphericalNFWPotential)

mw_potential = gp.CCompositePotential()
mw_potential['nucl'] = GrowingHernquistPotential(m0=2E9, c0=0.1, units=galactic)
mw_potential['bulge'] = GrowingHernquistPotential(m0=5E9, c0=1., units=galactic)
mw_potential['disk'] = GrowingMiyamotoNagaiPotential(m0=6.8E10*u.Msun, a0=3*u.kpc, b=280*u.pc,
                                                     units=galactic)
mw_potential['halo'] = GrowingSphericalNFWPotential(m0=6E11, r_s=16., units=galactic)

