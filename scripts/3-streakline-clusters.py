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

from uncluster import OutputPaths
paths = OutputPaths()
from uncluster.config import t_evolve, mw_potential
from uncluster.cluster_massloss import solve_mass_radius

def main(overwrite=False):

    # TODO: choose DF class at command line??
    df_name = "sph_iso"

    if not exists(paths.gc_w0.format(df_name)):
        raise IOError("File '{}' does not exist -- have you run 1-make-cluster-props.py?"
                      .format(paths.gc_w0.format(df_name)))

    t_grid = np.linspace(0., t_evolve.to(u.Gyr).value, 4096)

    with h5py.File(paths.gc_w0.format(df_name), 'r') as f:
        n_clusters = f.attrs['n']

        gc_mass = f['mass'][:]
        gc_radius = np.sqrt(np.sum(f['w0_pos'][:]**2, axis=0))
        ecc = f['ecc'][:]

        w0 = gd.CartesianPhaseSpacePosition(pos=f['w0_pos'][:]*u.kpc,
                                            vel=f['w0_vel'][:]*u.kpc/u.Myr)

    disrupt_idx = np.zeros(n_clusters).astype(int)
    final_m = np.zeros(n_clusters)
    final_r = np.zeros(n_clusters)
    all_m = []
    for i in range(n_clusters):
        if np.isnan(ecc[i]): continue
        idx, m, r = solve_mass_radius(gc_mass[i], gc_radius[i],
                                      ecc[i], t_grid)
        disrupt_idx[i] = idx[0]

        # mass and density distribution of surviving clusters
        final_m[i] = m[0][idx[0]+1]
        final_r[i] = r[0][idx[0]+1]

        all_m.append(m[0])

    # set disruption time to NaN for those that don't disrupt
    t_disrupt = t_grid[disrupt_idx+1]
    t_disrupt[(t_disrupt == t_evolve.to(u.Gyr).value) | (t_disrupt == 0)] = np.nan

    logger.info("{}/{} clusters survived".format(np.isnan(t_disrupt).sum(), len(t_disrupt)))

    # plot the disruption times
    fig,ax = plt.subplots(1,1)
    ax.hist(t_disrupt[np.isfinite(t_disrupt)], bins=np.linspace(0,11.5,12))
    ax.set_yscale('log')
    ax.set_title("{}/{} clusters survived".format(np.isnan(t_disrupt).sum(), len(t_disrupt)))
    ax.set_xlabel(r'$t_{\rm disrupt}$ [Gyr]')
    fig.savefig(join(paths.plot, 'disruption-times-{}.pdf'.format(df_name)))

    fig,axes = plt.subplots(1,2,figsize=(12,6))

    axes[0].hist(gc_mass, bins=np.logspace(4,7.1,9), alpha=0.3)
    axes[0].hist(final_m[np.isnan(t_disrupt)], bins=np.logspace(3,7.1,12), alpha=0.3)

    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlim(1E3, 3E7)
    axes[0].set_ylim(5E-1, 1E4)
    axes[0].set_xlabel(r"Mass [${\rm M}_\odot$]")
    axes[0].set_ylabel(r"$N$")

    # read data from harris GC catalog
    harris_filename = get_pkg_data_filename('data/harris-gc-catalog.fits',
                                            package='uncluster')
    harris_data = fits.getdata(harris_filename, 1)
    c = coord.SkyCoord(ra=harris_data['RA']*u.degree, dec=harris_data['DEC']*u.degree,
                       distance=harris_data['HELIO_DISTANCE']*u.kpc)
    gc = c.transform_to(coord.Galactocentric)
    harris_gc_distance = np.sqrt(np.sum(gc.cartesian.xyz**2, axis=0))
    harris_gc_distance = harris_gc_distance.to(u.kpc).value

    bins = np.logspace(-1.,2.,16)
    H,_ = np.histogram(gc_radius, bins=bins)
    data_H,_ = np.histogram(harris_gc_distance, bins=bins)

    V = 4/3*np.pi*(bins[1:]**3 - bins[:-1]**3)
    bin_cen = (bins[1:]+bins[:-1])/2.
    axes[1].plot(bin_cen, H/V, ls='--', marker=None)
    axes[1].errorbar(bin_cen, data_H/V, np.sqrt(data_H)/V,
                     color='k', marker='o', ecolor='#666666', linestyle='none')

    H_f,_ = np.histogram(final_r[np.isnan(t_disrupt)], bins=bins)
    axes[1].plot(bin_cen, H_f/V, ls='--', marker=None)

    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlim(1E-1, 1E2)
    axes[1].set_ylim(1E-7, 1E2)
    axes[1].set_xlabel(r"$r$ [kpc]")
    axes[1].set_ylabel('GC density [kpc$^{-3}$]')

    fig.tight_layout()
    fig.savefig(join(paths.plot, 'initial-final-density-{}.pdf'.format(df_name)))

    # ---------------------------------------------------------
    # Now we'll run some streakline models:

    for i in range(n_clusters)[1:]:
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

        # plt.figure()
        # gc_orbit.plot()
        # plt.show()

        # time grid of all_m[i] and gc_orbit are DIFFERENT - interpolate
        mass_interp_func = interp1d(t_grid[:disrupt_idx[i]],
                                    all_m[i][:disrupt_idx[i]],
                                    fill_value='extrapolate')

        m_t = mass_interp_func(gc_orbit.t.to(u.Gyr).value)
        m_t[m_t<0] = 0.

        release_every = 4 # MAGIC NUMBER
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

        release_time = np.vstack((gc_orbit.t[::release_every].to(u.Myr).value,
                                  gc_orbit.t[::release_every].to(u.Myr).value)).T.ravel()
        idx = (release_time*u.Myr) < (t_disrupt[i]*u.Gyr)

        plt.figure()
        stream[idx].plot(alpha=0.1)
        plt.show()

        break

    # # write to hdf5 file
    # with h5py.File(output_filename, 'w') as f:
    #     f.create_dataset('time', data=t_grid)
    #     f.attrs['n'] = len(t_disrupt)

    #     cl = f.create_group('clusters')
    #     for i in range(len(gc_props)):
    #         # set mass to 0 at disruption index
    #         if not np.isnan(t_disrupt[i]):
    #             m[i][disrupt_idx[i]] = 0.

    #         g = cl.create_group(str(i))
    #         g.create_dataset('mass', data=m[i][:disrupt_idx[i]+1])
    #         g.create_dataset('radius', data=r[i][:disrupt_idx[i]+1])
    #         g.attrs['t_disrupt'] = t_disrupt[i]
    #         g.attrs['disrupt_idx'] = disrupt_idx[i]

if __name__ == '__main__':
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")

    parser.add_argument("-s", "--seed", dest="seed", default=8675309,
                        type=int, help="Random number generator seed.")
    parser.add_argument("-o", "--overwirte", action="store_true", dest="overwrite",
                        default=False, help="Destroy everything.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    np.random.seed(args.seed)
    main(args.overwrite)
