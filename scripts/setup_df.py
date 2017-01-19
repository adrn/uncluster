"""
    This script generates a grid of Energy-DF values and caches the grid to a CSV file. The grid is
    used to inverse-transform sample energies from the DF, which are eventually turned into orbital
    initial conditions by later scripts.
"""

# Standard library
import sys

# Third-party
from astropy.table import QTable
import astropy.units as u
from gala.units import galactic
import numpy as np
from schwimmbad import choose_pool

# from uncluster.cluster_distributions.gnedin import sample_radii, sample_masses
from uncluster.log import logger
from uncluster.config import t_max
from uncluster.potential import mw_potential
from uncluster.cluster_distributions.apw import gc_prob_density
from uncluster.distribution_function import SphericalIsotropicDF
from uncluster.paths import Paths
paths = Paths()

def main(pool, df_name="sph_iso", overwrite=False):
    # TODO: right now df_name does nothing - specify df name at command line?

    # cache filename
    interp_grid_path = paths.cache / "interp_grid_{}.ecsv".format(df_name)

    # Now we need to evaluate the log(df) at a grid of energies so we can
    #   sample velocities.
    df = SphericalIsotropicDF(tracer=gc_prob_density,
                              background=mw_potential,
                              time=t_max)

    if interp_grid_path.exists() and overwrite:
        interp_grid_path.unlink()

    # The first thing we need to do is generate a grid of energy values and compute
    #   the value of the DF at these energies. We can than use inverse-transform
    #   sampling to sample energies from the DF to generate initial conditions.
    if not interp_grid_path.exists() or overwrite:
        logger.debug("DF interpolation grid file not found - generating ({})"
                     .format(str(interp_grid_path)))

        # generate a grid of energies to evaluate the DF on
        n_grid = 1024 # MAGIC NUMBER
        r = np.array([[1E-4,0,0],
                      [1E3,0,0]]).T * u.kpc
        v = mw_potential.circular_velocity(r, t=t_max)
        E_min = mw_potential.value(r[:,0], t=t_max).decompose(galactic).value[0]
        E_max = (mw_potential.value(r[:,1], t=t_max) + 0.5*v[1]**2).decompose(galactic).value[0]
        E_grid = np.linspace(E_min, E_max, n_grid)

        df.compute_ln_df_grid(E_grid, pool)

        tbl = QTable({'energy': df._energy_grid * galactic['energy']/galactic['mass'],
                      'log_df': df._log_df_grid})
        tbl.write(str(interp_grid_path), format='ascii.ecsv')

    else:
        logger.debug("DF interpolation grid file already exists - skipping ({})"
                     .format(str(interp_grid_path)))

    pool.close()
    sys.exit(0)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0, dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0, dest='quietness')

    parser.add_argument('-s', '--seed', dest='seed', default=None,
                        type=int, help='Random number generator seed.')
    parser.add_argument('-o', '--overwrite', action='store_true', dest='overwrite',
                        default=False, help='Destroy everything.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--ncores', dest='n_cores', default=1,
                       type=int, help='Number of CPU cores to use.')
    group.add_argument('--mpi', dest='mpi', default=False,
                       action='store_true', help='Run with MPI.')

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbosity != 0:
        if args.verbosity == 1:
            logger.setLevel(logging.DEBUG)
        else: # anything >= 2
            logger.setLevel(1)

    elif args.quietness != 0:
        if args.quietness == 1:
            logger.setLevel(logging.WARNING)
        else: # anything >= 2
            logger.setLevel(logging.ERROR)

    else: # default
        logger.setLevel(logging.INFO)

    if args.seed is not None:
        np.random.seed(args.seed)

    pool = choose_pool(mpi=args.mpi, processes=args.n_cores)
    logger.debug("Using pool: {}".format(pool.__class__))

    main(pool=pool, overwrite=args.overwrite)
