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

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from os.path import join, exists

# Third-party
from astropy import log as logger
import astropy.coordinates as coord
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import astropy.units as u
import gala.integrate as gi
import gala.dynamics as gd
from gala.dynamics.mockstream import dissolved_fardal_stream, fardal_stream
from gala.units import galactic
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from schwimmbad import choose_pool

from uncluster.paths import Paths
paths = Paths()
from uncluster.config import t_evolve, mw_potential
from uncluster.cluster_massloss import solve_mass_radius

class MockStreamWorker(object):

    def __init__(self):
        pass

    def work(self):
        pass

    def __call__(self, args):
        args
        return self.work()

# TODO: specify df name at command line
def main(gc_properties_file, pool, overwrite=False):

    # Load
    if not exists(gc_properties_file):
        raise IOError("File '{}' does not exist -- have you run "
                      "1-make-clusters.py?".format(gc_properties_file))

    # Load the initial conditions
    with h5py.File(gc_properties_file, 'r') as f:
        n_clusters = f.attrs['n']

        gc_masses = f['mass'][:]
        gc_radii = np.sqrt(np.sum(f['w0_pos'][:]**2, axis=0))
        circuls = f['circularity'][:]

        w0 = gd.CartesianPhaseSpacePosition(pos=f['w0_pos'][:]*u.kpc,
                                            vel=f['w0_vel'][:]*u.kpc/u.Myr)

    # Evolve the masses of the clusters by solving dM/dt using the prescription
    #   in Gnedin et al. 2014
    t_grid = np.linspace(0., t_evolve.to(u.Gyr).value, 4096)
    disrupt_idx = np.zeros(n_clusters).astype(int)
    final_m = np.zeros(n_clusters)
    final_r = np.zeros(n_clusters)
    all_m = []
    for i in range(n_clusters):
        if np.isnan(circuls[i]): # if circularity is NaN, skip
            logger.debug("Skipping cluster {} because circularity is NaN".format(i))
            continue

        # actually solve dM/dt
        try:
            idx, m, r = solve_mass_radius(gc_masses[i], gc_radii[i],
                                          circuls[i], t_grid)
        except Exception as e:
            logger.error("Failed to solve for mass-loss histor for cluster {}: \n\t {}"
                         .format(i, e.message))
            continue

        disrupt_idx[i] = idx[0]

        # mass and radial profile of surviving clusters
        final_m[i] = m[0][idx[0]+1]
        final_r[i] = r[0][idx[0]+1]

        all_m.append(m[0])

    # set disruption time to NaN for those that don't disrupt
    t_disrupt = t_grid[disrupt_idx+1]
    t_disrupt[(t_disrupt == t_evolve.to(u.Gyr).value) | (t_disrupt == 0)] = np.nan

    logger.info("{}/{} clusters survived".format(np.isnan(t_disrupt).sum(), len(t_disrupt)))

    # ---------------------------------------------------------
    # Now generate mock streams along the orbits

    for i in range(n_clusters):
        one_w0 = w0[i]

        # r = np.sqrt(np.sum(one_w0.pos**2))
        # v = np.sqrt(np.sum(one_w0.vel**2))
        # t_cross = r / v
        # dt = t_cross / 1024 # MAGIC NUMBER
        dt = 1.*u.Myr # MAGIC NUMBER
        gc_orbit = mw_potential.integrate_orbit(one_w0, dt=dt,
                                                t1=0.*u.Gyr, t2=11.5*u.Gyr,
                                                Integrator=gi.DOPRI853Integrator)
        logger.debug("Orbit integrated for {} steps".format(len(gc_orbit.t)))

        # don't make a stream if its final radius is outside of the virial radius
        if np.sqrt(np.sum(gc_orbit.pos[-1]**2)) > 500*u.kpc:
            logger.debug("Cluster {} ended up way outside of the virial radius. "
                         "Not making a mock stream".format(i))
            continue

        # time grid of all_m[i] and gc_orbit are DIFFERENT - interpolate
        mass_interp_func = interp1d(t_grid[:disrupt_idx[i]],
                                    all_m[i][:disrupt_idx[i]],
                                    fill_value='extrapolate')

        m_t = mass_interp_func(gc_orbit.t.to(u.Gyr).value)
        m_t[m_t<=0] = 1. # HACK: can mock_stream not handle m=0?

        release_every = 4 # MAGIC NUMBER
        # TODO: can weight each particle by (4*dt) * (dm/dt) -- the amount of mass in that particle

        logger.debug("Generating mock stream with {} particles"
                     .format(len(gc_orbit.t)//release_every*2))

        if np.isnan(t_disrupt[i]): # cluster doesn't disrupt
            logger.debug("Cluster didn't disrupt")
            stream = fardal_stream(mw_potential, gc_orbit, m_t*u.Msun,
                                   release_every=release_every,
                                   Integrator=gi.DOPRI853Integrator)

        else: # cluster disrupts completely
            logger.debug("Cluster fully disrupted at {}".format(t_disrupt[i]*u.Gyr))
            stream = dissolved_fardal_stream(mw_potential, gc_orbit, m_t*u.Msun,
                                             t_disrupt[i]*u.Gyr, release_every=release_every,
                                             Integrator=gi.DOPRI853Integrator)
        logger.debug("Done generating mock stream.")

        release_time = np.vstack((gc_orbit.t[::release_every].to(u.Myr).value,
                                  gc_orbit.t[::release_every].to(u.Myr).value)).T.ravel()
        idx = (release_time*u.Myr) < (t_disrupt[i]*u.Gyr)

        plt.figure()
        stream[idx].plot(alpha=0.1)
        plt.show()

        break

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

    parser.add_argument('-f', '--gc-props-file', dest='gc_properties', required=True,
                        type=str, help='Path to the cache file containing cluster properties.')

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
    main(gc_properties_file=args.gc_properties,
         pool=pool, overwrite=args.overwrite)
