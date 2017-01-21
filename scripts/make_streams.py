"""
    Solve for the mass-loss rates and destruction times for each cluster given
    its eccentricity, initial mass, and initial mean orbital radius. Then,
    generate Streakline+scatter+full-disruption simulations for each globular
    cluster stream.

    This code makes the following simplifying assumptions:
        - Dynamical friction is neglected for the streakline models: we are
          interested in clusters with r > 5 kpc, where dynamical friction has
          a negligible effect.
        -

"""

__author__ = "adrn <adrn@astro.princeton.edu>"

# Standard library
from os.path import join, exists
import sys
import time
from pathlib import Path

# Third-party
from astropy import log as logger
import astropy.units as u
import gala.integrate as gi
import gala.dynamics as gd
from gala.dynamics.mockstream import dissolved_fardal_stream, fardal_stream
import gala.potential as gp
from gala.units import galactic
import h5py
import numpy as np
from scipy.interpolate import interp1d
from schwimmbad import choose_pool

# Project
from uncluster.config import t_max
from uncluster.potential import mw_potential
from uncluster.cluster_massloss import solve_mass_evolution
from uncluster.paths import Paths
paths = Paths()

class MockStreamWorker(object):

    def __init__(self, cache_file, t_grid, overwrite=False, release_every=4): # MAGIC DEFAULT
        self.cache_file = cache_file
        self.t_grid = t_grid
        self.overwrite = overwrite
        self.release_every = release_every
        self.H = gp.Hamiltonian(mw_potential)

    def work(self, i, initial_mass, w0, dt):

        with h5py.File(str(self.cache_file), 'r') as root:
            if str(i) in root['mock_streams'] and not self.overwrite:
                logger.debug("Cluster {} already done.".format(i))
                return

        logger.debug("Cluster {} initial mass = {:.0e} Msun"
                     .format(i, initial_mass))

        # TODO: integrate orbit
        gc_orbit = self.H.integrate_orbit(w0, dt=dt,
                                          t1=t_max, t2=0.,
                                          Integrator=gi.DOPRI853Integrator)
        logger.debug("\t Orbit integrated for {} steps".format(len(gc_orbit.t)))

        t_grid = gc_orbit.t
        r_grid = gc_orbit.r

        # solve dM/dt to get mass-loss history
        try:
            disrupt_idx, mass_grid = solve_mass_evolution(initial_mass, t_grid.to(u.Gyr).value,
                                                          r_grid.to(u.kpc).value)
        except:
            logger.error("Failed to solve for mass-loss history for cluster {}: \n\t {}"
                         .format(i, sys.exc_info()[0]))
            return

        # set disruption time to NaN for those that don't disrupt
        t_disrupt = t_grid[disrupt_idx+1]
        if (t_disrupt > -1*u.Myr):
            t_disrupt = np.nan # didn't disrupt
            logger.debug("\t Cluster survived!")
        else:
            logger.debug("\t Cluster disrupted at: {}".format(t_disrupt))

        # don't make a stream if its final radius is outside of the virial radius
        if np.sqrt(np.sum(gc_orbit.pos[:,-1]**2)) > 500*u.kpc:
            r0 = np.sqrt(np.sum(w0.pos**2))
            v0 = np.sqrt(np.sum(w0.vel**2))
            logger.debug("Cluster {} ended up way outside of the virial radius. "
                         "r0={}, v0={}."
                         "Not making a mock stream".format(i, r0, v0))
            return

        # orbit has different times than mass_grid
        # TODO: no longer true!
        mass_interp_func = interp1d(t_grid.to(u.Myr), mass_grid, fill_value='extrapolate')
        m_t = mass_interp_func(gc_orbit.t.to(u.Myr).value)
        m_t[m_t<=0] = 1. # Msun HACK: can mock_stream not handle m=0?
        m_t = mass_grid

        logger.debug("\t Generating mock stream with {} particles over {} steps"
                     .format(len(gc_orbit.t)//self.release_every*2, len(gc_orbit.t)))

        _timer0 = time.time()
        try:
            if np.isnan(t_disrupt): # cluster doesn't disrupt
                stream = fardal_stream(self.H, gc_orbit, m_t*u.Msun,
                                       release_every=self.release_every,
                                       Integrator=gi.DOPRI853Integrator)

            else: # cluster disrupts completely
                stream = dissolved_fardal_stream(self.H, gc_orbit, m_t*u.Msun,
                                                 t_disrupt, release_every=self.release_every,
                                                 Integrator=gi.DOPRI853Integrator)
        except:
            logger.error("\t Failed to generate mock stream for cluster {}: \n\t {}"
                         .format(i, sys.exc_info()[0]))
            return

        logger.debug("\t ...done generating mock stream ({:.2f} seconds)."
                     .format(time.time()-_timer0))

        release_time = np.vstack((gc_orbit.t[::self.release_every].to(u.Myr).value,
                                  gc_orbit.t[::self.release_every].to(u.Myr).value)).T.ravel()
        idx = (release_time*u.Myr) < (t_disrupt*u.Myr)

        # get dm/dt at each release_time
        h = dt.to(u.Myr).value
        dM_dt = (mass_interp_func(release_time+h) - mass_interp_func(release_time)) / h

        release_time_dt = self.release_every * dt.to(u.Myr).value

        # can weight each particle by release_times * (dm/dt) -- the amount of mass in that particle
        particle_weights = -dM_dt * release_time_dt * 0.5 # mass lost split btwn L pts

        return i, t_disrupt, stream[idx], particle_weights[idx]

    def __call__(self, args):
        return self.work(*args)

    def callback(self, result):
        if result is None:
            pass

        else:
            i, t_disrupt, stream, particle_weights = result

            with h5py.File(str(self.cache_file), 'a') as root:
                f = root['mock_streams']

                if str(i) in f:
                    del root['mock_streams'][str(i)]

                g = f.create_group(str(i))
                g.attrs['t_disrupt'] = t_disrupt

                d = g.create_dataset('stream_pos', data=stream.pos.value)
                d.attrs['unit'] = str(stream.pos.unit)

                d = g.create_dataset('stream_vel', data=stream.vel.value)
                d.attrs['unit'] = str(stream.vel.unit)

                d = g.create_dataset('stream_weights', data=particle_weights)
                d.attrs['unit'] = str(u.Msun)

# TODO: specify df name at command line
def main(cache_file, pool, overwrite=False):
    cache_file = Path(cache_file).expanduser().absolute()

    if not cache_file.exists():
        raise IOError("File '{}' does not exist -- have you run "
                      "make_clusters.py?".format(str(cache_file)))

    else:
        with h5py.File(str(cache_file), 'r') as f:
            if 'cluster_properties' not in f:
                raise IOError("File '{}' does not contain a 'cluster_properties' group "
                              "-- have you run make_clusters.py?".format(str(cache_file)))

    # Load the initial conditions
    with h5py.File(str(cache_file), 'a') as root:
        f = root['cluster_properties']
        n_clusters = f.attrs['n']

        gc_masses = f['mass'][:]
        w0 = gd.CartesianPhaseSpacePosition(pos=f['w0_pos'][:]*u.kpc,
                                            vel=f['w0_vel'][:]*u.kpc/u.Myr)

        if 'mock_streams' not in root:
            root.create_group('mock_streams')

    # Used to evolve the masses of the clusters by solving dM/dt using the prescription
    #   in Gnedin et al. 2014 (in worker above)
    t_grid = np.linspace(t_max.to(u.Myr).value, 0., 4096) # MAGIC NUMBER

    # Used for orbit integration. For now, all have same value, might want to adapt to
    #   the crossing time or something...
    dt = 1*u.Myr

    worker = MockStreamWorker(t_grid=t_grid,
                              cache_file=cache_file,
                              overwrite=overwrite,
                              release_every=4) # MAGIC NUMBER
    tasks = [[i, gc_masses[i], w0[i], dt] for i in range(n_clusters)]

    for r in pool.map(worker, tasks, callback=worker.callback):
        pass

    pool.close()
    sys.exit(0)

if __name__ == '__main__':
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0, dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0, dest='quietness')

    parser.add_argument('-o', '--overwrite', action='store_true', dest='overwrite',
                        default=False, help='Destroy everything.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--ncores', dest='n_cores', default=1,
                       type=int, help='Number of CPU cores to use.')
    group.add_argument('--mpi', dest='mpi', default=False,
                       action='store_true', help='Run with MPI.')

    parser.add_argument('-f', '--cache-file', dest='cache_file', required=True,
                        type=str, help='Path to the cache file which should already '
                                       'contain the cluster properties.')

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

    pool = choose_pool(mpi=args.mpi, processes=args.n_cores)
    logger.info("Using pool: {}".format(pool.__class__))

    main(cache_file=args.cache_file, pool=pool, overwrite=args.overwrite)
