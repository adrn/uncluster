# Third-party
from gala.units import galactic

# Project
from .components import GrowingMilkyWayPotential
from .potential_config import m_h, r_s, m_n, c_n

# best-fit values come from running TODO:
mw_potential = GrowingMilkyWayPotential(m_n0=m_n, c_n0=c_n,
                                        m_h0=m_h, r_s=r_s,
                                        units=galactic)
