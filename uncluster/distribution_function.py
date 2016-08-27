# Standard library
from inspect import signature

# Third-party
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

    def __init__(self, tracer, background, energy_grid=None):
        """
        Parameters
        ----------
        tracer : callable, :class:`~gala.potential.PotentialBase`
            This can either be (1) a function / callable that accepts a value of the *potential*
            as its radial coordinate (see Binney & Tremaine, Chapter 4) and computes the density,
            or (2) an instance of a :class:`~gala.potential.PotentialBase` subclass.
        background : callable, :class:`~gala.potential.PotentialBase`
            This can either be (1) a function / callable that accepts a single radius, or (2) an
            instance of a :class:`~gala.potential.PotentialBase` subclass.
        energy_grid : array_like (optional)
            Array of energies to compute the DF along. If not provided, it's assumed you'll pass
            your own energy grid and DF values to: `SphericalIsotropicDF.make_ln_df_interp_func`.

        TODO: actually, this is borked. tracer has to be a function -- either takes phi and all is
        good, or takes r and has to be a probability density!
        """

        if isinstance(background, gp.PotentialBase):
            self._bg_potential = lambda r: float(background._value(np.array([[r,0.,0.]]).T))
        else:
            self._bg_potential = background
            self._validate_rad_func(self._bg_potential)

        if isinstance(tracer, gp.PotentialBase):
            logger.debug("SphericalIsotropicDF: need to solve for r(phi) -- this will slow down "
                         "generating the DF grid.")
            self._density_phi = lambda phi: float(tracer._density(np.array([[self._r_from_phi(phi),0.,0.]]).T))
        else:
            self._validate_rad_func(tracer)

            if 'phi' in signature(tracer).parameters:
                self._density_phi = tracer
            elif 'r' in signature(tracer).parameters:
                self._density_phi = lambda phi: tracer(self._r_from_phi(phi))
            else:
                raise ValueError("Invalid tracer density function") # TODO: why?

        # TODO: some way to automatically estimate energy_grid...
        # if energy_grid is None:
        #     curlyE = np.linspace(1E-2, 1.1, 256) # MAGIC NUMBER

        #     # estimate typical energy
        #     energy_grid = -curlyE * E_scale**2
        if energy_grid is not None:
            self._make_ln_df_grid(energy_grid)

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

    def _make_ln_df_grid(self, E_grid):
        df = np.zeros(len(E_grid))
        for i,energy in enumerate(E_grid):
            df[i] = derivative(lambda H: quad(self.eddington_integrand, H, 0, args=(H,))[0],
                               energy, dx=np.abs(1E-4*energy))
        log_df = np.log(df)
        idx = np.isfinite(log_df)

        if (idx.sum() / len(idx)) < 0.75:
            logger.warning("Less than 75% of the DF values along the supplied energy "
                           "grid have finite values. Consider changing the bounds of "
                           "the energy grid for better sampling and more reliable "
                           "interpolation.")

        self._energy_grid = E_grid[idx]
        self._df_grid = df[idx]
        self.make_ln_df_interp_func(E_grid[idx], log_df[idx])

    def make_ln_df_interp_func(self, E_grid, log_df_grid):
        self._energy_grid = E_grid
        self._df_grid = np.exp(log_df_grid)
        self.log_df_interp = interp1d(E_grid, log_df_grid, fill_value="extrapolate")

    def ln_f_v2(self, v, r):
        if v <= 0.:
            return -np.inf

        E = 0.5*v**2 + self._bg_potential(r)
        return self.log_df_interp(E) + 2*np.log(v)
