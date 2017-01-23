"""
    TODO: explain
"""

# Standard library
import sys

# Third-party
from astropy.table import QTable
import astropy.units as u
import gala.dynamics as gd
from gala.units import galactic
import h5py
import numpy as np
from schwimmbad import choose_pool

# from uncluster.cluster_distributions.gnedin import sample_radii, sample_masses
from uncluster.log import logger
from uncluster.cluster_distributions.apw import sample_radii, sample_masses
from uncluster.config import t_max
from uncluster.potential import mw_potential
from uncluster.cluster_distributions.apw import gc_prob_density
from uncluster.distribution_function import SphericalIsotropicDF
from uncluster.paths import Paths
from uncluster.utils import quantity_to_hdf5
paths = Paths()

def v_worker(task):
    df,r,n_samples = task

    # rejection sample to get velocity
    vs = np.random.uniform(0, 0.5, n_samples) # MAGIC NUMBER: 0.5 (max velocity in kpc/Myr)
    ll = np.array([df.ln_f_v2(v, r) for v in vs])
    uu = np.random.uniform(size=n_samples)
    vs = vs[uu < np.exp(ll - ll.max())]

    if len(vs) <= 1:
        logger.warning("Rejection sampling returned <=1 samples - raise n_samples")
        return np.nan
    i = np.random.randint(len(vs))
    return vs[i]

def main(pool, df_name="sph_iso", overwrite=False):
    # TODO: right now df_name does nothing - specify df name at command line?

    # cache filenames
    cache_file_path = paths.cache / "{}.hdf5".format(df_name)
    interp_grid_path = paths.cache / "interp_grid_{}.ecsv".format(df_name)

    if not interp_grid_path.exists():
        raise IOError("DF interpolation grid not found at {} -- run setup_df.py first."
                      .format(str(interp_grid_path)))

    # Now we need to evaluate the log(df) at a grid of energies so we can
    #   sample velocities.
    df = SphericalIsotropicDF(tracer=gc_prob_density,
                              background=mw_potential,
                              time=t_max)

    # interpolation grid cached
    logger.debug("Reading DF interpolation grid from {}".format(str(interp_grid_path)))
    tbl = QTable.read(str(interp_grid_path), format='ascii.ecsv')
    energy_grid = tbl['energy'].decompose(galactic).value
    log_df_grid = tbl['log_df']
    df.make_ln_df_interp_func(energy_grid, log_df_grid)

    if cache_file_path.exists():
        with h5py.File(str(cache_file_path), 'r+') as f:
            if 'progenitors' in f and not overwrite:
                logger.info("Cache file {:} already contains results. Use --overwrite "
                            "to overwrite.".format(str(cache_file_path)))
                pool.close()
                sys.exit(0)

            elif 'progenitors' in f and overwrite:
                del f['progenitors']

    else:
        with h5py.File(str(cache_file_path), 'w') as f:
            pass

    # The actual number here is arbitrary because we later post-process to get the
    #   properties of the initial population from the final population
    n_clusters = 10000 # MAGIC NUMBER
    gc_mass = sample_masses(size=n_clusters)
    logger.info("Sampled {} cluster masses (M_tot = {:.2e})".format(n_clusters, gc_mass.sum()))

    # only take radii out to ~virial radius
    logger.debug("Sampling cluster radii...")
    gc_radius = sample_radii(r_min=0.1, r_max=250., size=n_clusters)
    logger.debug("...done.")

    n_samples = 100
    tasks = [(df,r,n_samples) for r in gc_radius.decompose(galactic).value]

    logger.debug("Sampling cluster velocities...")
    results = [r for r in pool.map(v_worker, tasks)]
    v_mag = np.array(results) * u.kpc/u.Myr
    logger.debug("...done.")

    if np.any(np.isnan(v_mag)):
        logger.warning("Failed to find velocities for {}/{} orbits."
                       .format(np.isnan(v_mag).sum(), n_clusters))

    q = np.zeros((3,n_clusters))
    q[0] = gc_radius
    E = 0.5*v_mag**2 + mw_potential.energy(q)
    if np.any(E.value[np.isfinite(E.value)] > 0):
        logger.warning("{} unbound orbits.".format((E > 0.).sum()))

    # need to turn the radius and velocity magnitude into 3D intial conditions
    pos,vel = df.r_v_to_3d(gc_radius[np.isfinite(v_mag)], v_mag[np.isfinite(v_mag)])
    w0 = gd.CartesianPhaseSpacePosition(pos=pos, vel=vel)

    n_bad = np.logical_not(np.isfinite(v_mag)).sum()
    if n_bad > 0:
        n_clusters -= n_bad
        logger.warning("Failed to get velocity / position for {} clusters".format(n_bad))

    pool.close()

    # Write out the initial conditions and cluster properties
    with h5py.File(str(cache_file_path), 'a') as root:
        f = root.create_group('progenitors')
        f.attrs['n'] = n_clusters

        w0_f = f.create_group('initial')
        quantity_to_hdf5(w0_f, 'pos', w0.pos)
        quantity_to_hdf5(w0_f, 'vel', w0.vel)
        quantity_to_hdf5(w0_f, 'mass', gc_mass)

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
