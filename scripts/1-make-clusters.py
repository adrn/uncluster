"""
    TODO: explain
"""

__author__ = "adrn <adrn@princeton.edu>"

# Standard library
from os.path import join, exists
import sys

# Third-party
from astropy import log as logger
from astropy.table import QTable
import astropy.units as u
import gala.integrate as gi
import gala.dynamics as gd
from gala.units import galactic
import h5py
import numpy as np
from scipy.optimize import minimize
from schwimmbad import choose_pool

from uncluster.paths import Paths
paths = Paths()

# from uncluster.cluster_distributions.gnedin import sample_radii, sample_masses
from uncluster.cluster_distributions.apw import sample_radii, sample_masses
from uncluster.config import f_gc, M_tot

from uncluster.config import mw_potential
from uncluster.cluster_distributions.apw import gc_prob_density
from uncluster.distribution_function import SphericalIsotropicDF

def v_worker(task):
    df,r,n_samples = task

    # rejection sample to get velocity
    vs = np.random.uniform(0, 0.5, n_samples) # MAGIC NUMBER 0.5 (max velocity)
    ll = np.array([df.ln_f_v2(v, r) for v in vs])
    uu = np.random.uniform(size=n_samples)
    vs = vs[uu < np.exp(ll - ll.max())]

    if len(vs) <= 1:
        raise ValueError("Rejection sampling returned <=1 samples - raise n_samples")
    i = np.random.randint(len(vs))
    return vs[i]

# TODO: specify df name at command line?
def main(pool, df_name="sph_iso", overwrite=False):

    # cache filenames
    gc_cache_path = join(paths.cache, "gc-properties-{}").format(df_name)
    interp_grid_path = join(paths.cache, "interp_grid_{}.ecsv").format(df_name)

    # - Sample masses until the total mass in GCs is equal to a fraction (f_gc) of the
    #   total mass in stars (M_tot):
    maxiter = 64000 # MAGIC NUMBER
    gc_mass = sample_masses(size=maxiter)
    for i in range(maxiter):
        _sum = gc_mass[:i+1].sum()
        if _sum >= f_gc*M_tot:
            break

    if i == (maxiter-1):
        raise ValueError("Reached maximum number of iterations when sampling masses.")

    gc_mass = gc_mass[:i+1]
    n_clusters = len(gc_mass)
    logger.info("Sampled {} cluster masses (M_tot = {:.2e})".format(n_clusters, gc_mass.sum()))

    # only take radii out to ~virial radius
    gc_radius = sample_radii(r_max=250., size=n_clusters)

    # Now we need to evaluate the log(df) at a grid of energies so we can
    #   sample velocities.
    df = SphericalIsotropicDF(tracer=gc_prob_density,
                              background=mw_potential)
    if not exists(interp_grid_path) or overwrite:
        # generate a grid of energies to evaluate the DF on
        n_grid = 1024 # MAGIC NUMBER
        r = np.array([[1E-4,0,0],
                      [1E3,0,0]]).T * u.kpc
        v = mw_potential.circular_velocity(r)
        E_min = mw_potential.value(r[:,0]).decompose(galactic).value[0]
        E_max = (mw_potential.value(r[:,1]) + 0.5*v[1]**2).decompose(galactic).value[0]
        E_grid = np.linspace(E_min, E_max, n_grid)

        df.compute_ln_df_grid(E_grid, pool)

        tbl = QTable({'energy': df._energy_grid * galactic['energy']/galactic['mass'],
                      'log_df': df._log_df_grid})
        tbl.write(interp_grid_path, format='ascii.ecsv')

    else:
        # interpolation grid already cached
        tbl = QTable.read(str(interp_grid_path), format='ascii.ecsv')
        energy_grid = tbl['energy'].decompose(galactic).value
        log_df_grid = tbl['log_df']

        df.make_ln_df_interp_func(energy_grid, log_df_grid)

    n_samples = 100
    tasks = [(df,r,n_samples) for r in gc_radius.decompose(galactic).value]

    results = pool.map(v_worker, tasks)
    v_mag = np.array(results) * u.kpc/u.Myr

    if np.any(np.isnan(v_mag)):
        logger.warning("Failed to find velocities for {}/{} orbits."
                       .format(np.isnan(v_mag).sum(), n_clusters))

    q = np.zeros((3,n_clusters))
    q[0] = gc_radius
    E = 0.5*v_mag**2 + mw_potential.potential(q)
    if np.any(E.value[np.isfinite(E.value)] > 0):
        logger.warning("{} unbound orbits.".format((E > 0.).sum()))

    # need to turn the radius and velocity magnitude into 3D intial conditions
    pos,vel = df.r_v_to_3d(gc_radius, v_mag)
    w0 = gd.CartesianPhaseSpacePosition(pos=pos, vel=vel)

    # compute circularities and final radii for the orbits
    t_cross = gc_radius / v_mag

    dt = t_cross / 256.
    n_steps = 8192 # 32 crossing times

    r_f = np.zeros_like(t_cross.value)
    J_Jc = np.zeros_like(t_cross.value)
    for i in range(n_clusters):
        logger.debug('Integrating {}, dt={:.3f}'.format(i, dt[i]))
        if np.isnan(dt[i]):
            r_f[i] = np.nan
            continue

        J = np.sqrt(np.sum(w0[i].angular_momentum()**2))
        Jc = np.sqrt(np.sum(w0[i].pos**2)) * mw_potential.circular_velocity(w0[i].pos)
        J_Jc[i] = (J / Jc).decompose()[0]

        w = mw_potential.integrate_orbit(w0[i], dt=dt[i], n_steps=n_steps,
                                         Integrator=gi.DOPRI853Integrator)
        r_f[i] = np.sqrt(np.sum(w.pos[:,-1]**2)).value

    idx = np.isfinite(J_Jc)
    if idx.sum() != len(idx):
        logger.warning("{}/{} failed circularities".format(n_clusters-idx.sum(),
                                                           n_clusters))

    # Write out the initial conditions and cluster properties
    with h5py.File(gc_cache_path, 'w') as f:
        f.attrs['n'] = n_clusters

        d = f.create_dataset('w0_pos', data=w0.pos)
        d.attrs['unit'] = 'kpc'

        d = f.create_dataset('w0_vel', data=w0.vel)
        d.attrs['unit'] = 'kpc/Myr'

        d = f.create_dataset('mass', data=gc_mass)
        d.attrs['unit'] = 'Msun'

        f.create_dataset('circularity', data=J_Jc)

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
    main(pool=pool, overwrite=args.overwrite)
