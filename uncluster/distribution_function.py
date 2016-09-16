# Standard library
from inspect import signature

# Third-party
import astropy.coordinates as coord
from astropy import log as logger
import astropy.units as u
import gala.potential as gp
import numpy as np
from scipy.misc import derivative
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import root

class DF(object):

    def _validate_rad_func(self, func):
        sig = signature(func)
        if len(sig.parameters) > 1:
            raise ValueError("The input function should take a single argument.")

class SphericalIsotropicDF(DF):

    def __init__(self, tracer, background):
        """
        Parameters
        ----------
        tracer : callable
            TODO: explain that name of argument matters!!
        background : callable, :class:`~gala.potential.PotentialBase`
            This can either be (1) a function / callable that accepts a single radius, or (2) an
            instance of a :class:`~gala.potential.PotentialBase` subclass.
        """

        self.background = background
        if isinstance(background, gp.PotentialBase):
            self._bg_potential = self._value_wrap
        else:
            self._bg_potential = background
            self._validate_rad_func(self._bg_potential)

        self._validate_rad_func(tracer)
        self.tracer = tracer
        if 'phi' in signature(tracer).parameters:
            self._density_phi = tracer
        elif 'r' in signature(tracer).parameters:
            self._density_phi = self._density_wrap
        else:
            raise ValueError("Invalid tracer density function") # TODO: why?

    def _value_wrap(self, r):
        return float(self.background._value(np.array([[r,0.,0.]]).T))

    def _density_wrap(self, phi):
        return self.tracer(self._r_from_phi(phi))

    def _r_from_phi(self, phi):
        # for x0 in np.concatenate(([10.], np.logspace(-2, 2, 8))):
        x0 = 1.
        res = root(lambda r: self._bg_potential(r[0]) - phi, x0)
        if res.success:
            return res.x[0]
        return np.nan

    def eddington_integrand(self, phi, H):
        dp_dphi = derivative(self._density_phi, phi, dx=1E-3*phi)
        return dp_dphi / np.sqrt(phi - H)

    def compute_ln_df_grid(self, E_grid, pool=None):
        """
        Parameters
        ----------
        E_grid : array_like (optional)
            Array of energies to compute the DF along. If not provided, it's assumed you'll pass
            your own energy grid and DF values to: `SphericalIsotropicDF.make_ln_df_interp_func`.
        """
        df = np.zeros(len(E_grid))
        for i,energy in enumerate(E_grid):
            df[i] = derivative(lambda H: quad(self.eddington_integrand, H, 0, args=(H,))[0],
                               energy, dx=np.abs(1E-4*energy))
        log_df = np.log(df / (np.sqrt(8.)*np.pi**2))
        idx = np.isfinite(log_df)

        if (idx.sum() / len(idx)) < 0.75:
            logger.warning("Less than 75% of the DF values along the supplied energy "
                           "grid have finite values. Consider changing the bounds of "
                           "the energy grid for better sampling and more reliable "
                           "interpolation.")
        else:
            logger.debug("{}/{} of the DF values are finite.".format(idx.sum(), len(idx)))

        self._energy_grid = E_grid[idx]
        self._log_df_grid = log_df[idx]
        self.make_ln_df_interp_func(E_grid[idx], log_df[idx])

    def make_ln_df_interp_func(self, E_grid, log_df_grid):
        self._energy_grid = E_grid
        self._log_df_grid = log_df_grid
        self.log_df_interp = interp1d(E_grid, log_df_grid,
                                      bounds_error=False, fill_value=-np.inf)

    def ln_f_v2(self, v, r):
        if v <= 0.:
            return -np.inf

        E = 0.5*v**2 + self._bg_potential(r)
        return self.log_df_interp(E) + 2*np.log(v)

    def r_v_to_3d(self, r, v):
        r = np.atleast_1d(r)
        v = np.atleast_1d(v)

        phi = np.random.uniform(0, 2*np.pi, size=r.size)
        theta = np.arccos(2*np.random.uniform(size=r.size) - 1)
        sph = coord.PhysicsSphericalRepresentation(phi=phi*u.radian, theta=theta*u.radian, r=r*u.one)
        xyz = sph.represent_as(coord.CartesianRepresentation).xyz

        phi_v = np.random.uniform(0, 2*np.pi, size=v.size)
        theta_v = np.arccos(2*np.random.uniform(size=v.size) - 1)
        v_sph = coord.PhysicsSphericalRepresentation(phi=phi_v*u.radian, theta=theta_v*u.radian, r=v*u.one)
        v_xyz = v_sph.represent_as(coord.CartesianRepresentation).xyz

        return xyz, v_xyz
