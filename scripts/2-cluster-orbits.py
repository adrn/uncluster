"""
    TODO: explain
"""

__author__ = "adrn <adrn@princeton.edu>"

# Standard library
from multiprocessing import Pool
from os.path import join, exists

# Third-party
from astropy import log as logger
from astropy.table import QTable
import astropy.units as u
import emcee
import gala.integrate as gi
import gala.dynamics as gd
from gala.units import galactic
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from uncluster.paths import Paths
paths = Paths()
from uncluster.config import mw_potential
from uncluster.cluster_distributions.apw import gc_prob_density
from uncluster.distribution_function import SphericalIsotropicDF

class Worker(object):

    def __init__(self, df, n_walkers=16):
        self.df = df
        self.n_walkers = n_walkers

    def work(self, i, m, r):
        # first, optimize to find a place to initialize walkers
        res = minimize(lambda v: -self.df.ln_f_v2(v, r), 0.1, method='powell')

        # print(m, r)
        # _vv = np.linspace(0, 0.3, 128)
        # plt.figure()
        # plt.plot(_vv, np.exp([self.df.ln_f_v2(vv, r) for vv in _vv]))
        # plt.show()

        if not res.success:
            logger.error("Failed to optimize for cluster {}!".format(i))
            return np.nan

        p0 = np.abs(np.random.normal(res.x, res.x*1E-2, (self.n_walkers,1)))
        sampler = emcee.EnsembleSampler(nwalkers=self.n_walkers, dim=1,
                                        lnpostfn=self.df.ln_f_v2, args=(r,))

        try:
            sampler.run_mcmc(p0, 128)
        except Warning:
            logger.error("Failed to MCMC cluster {}!".format(i))
            return np.nan

        # the randint is redundant, but just being safe...
        return sampler.chain[np.random.randint(self.n_walkers), -1, 0]

    def __call__(self, args):
        i,m,r = args
        return self.work(i, m, r)

def main(overwrite=False):
    if not exists(paths.gc_properties):
        raise IOError("File '{}' does not exist -- have you run 1-make-cluster-props.py?"
                      .format(paths.gc_properties))

    # read radii and masses from cached file
    gc_props = QTable.read(paths.gc_properties, format='ascii.ecsv')

    gc_mass = gc_props['mass'].to(u.Msun).value
    gc_radius = gc_props['radius'].to(u.kpc).value
    n_clusters = len(gc_props)

    # TODO: choose DF class at command line??
    df_name = "sph_iso"

    # filename to cache interpolation grid
    interp_grid_path = join(paths.cache, "interp_grid_{}.ecsv").format(df_name)
    if not interp_grid_path.exists() or overwrite:
        # generate a grid of energies to evaluate the DF on
        n_grid = 1024
        r = np.array([[1E-4,0,0],
                      [1E3,0,0]]).T * u.kpc
        v = mw_potential.circular_velocity(r)
        E_min = (mw_potential.value(r[:,0]) + 0.5*v**2).decompose(galactic).value[0]
        E_max = (mw_potential.value(r[:,1]) + 0.5*v**2).decompose(galactic).value[0]

        iso = SphericalIsotropicDF(tracer=gc_prob_density,
                                   background=mw_potential,
                                   energy_grid=np.linspace(E_min, E_max, n_grid))

        tbl = QTable({'energy': iso._energy_grid * galactic['energy']/galactic['mass'],
                      'df': iso._df_grid})
        tbl.write(str(interp_grid_path), format='ascii.ecsv')

    else:
        # interpolation grid already cached
        tbl = QTable.read(str(interp_grid_path), format='ascii.ecsv')
        energy_grid = tbl['energy'].decompose(galactic).value
        log_df_grid = np.log(tbl['df'])

        iso = SphericalIsotropicDF(tracer=gc_prob_density,
                                   background=mw_potential)
        iso.make_ln_df_interp_func(energy_grid, log_df_grid)

    # plot the interpolated DF
    fig,ax = plt.subplots(1,1,figsize=(6,4))
    ax.semilogy(-iso._energy_grid, np.exp(iso._log_df_grid))
    ax.set_xlabel(r'-E [${\rm kpc}^2 \, {\rm Myr}^{-2}$]')
    ax.set_ylabel("df")
    fig.tight_layout()
    fig.savefig(join(paths.plots, 'df-vs-energy-{}.pdf').format(df_name))

    # now I need to draw from the velocity distribution -- using emcee to do the sampling
    worker = Worker(df=iso, n_walkers=16)
    tasks = list(zip(range(n_clusters), gc_mass, gc_radius))

    # HACK: only do 256 for now for speed
    DERP = 512
    n_clusters = DERP
    gc_radius = gc_radius[:DERP]
    with Pool() as p: # use all CPUs
        v_mag = p.map(worker, tasks[:DERP])
    v_mag = np.array(v_mag)

    if np.any(np.isnan(v_mag)):
        logger.warning("Failed to find velocities for {}/{} orbits."
                       .format(np.isnan(v_mag).sum(), n_clusters))

    q = np.zeros((3,n_clusters))
    q[0] = gc_radius
    E = 0.5*(v_mag*u.kpc/u.Myr)**2 + mw_potential.potential(q)
    if np.any(E.value[np.isfinite(E.value)] > 0):
        logger.warning("{} unbound orbits.".format((E > 0.).sum()))

    # need to turn the radius and velocity magnitude into 3D intial conditions
    pos,vel = iso.r_v_to_3d(gc_radius, v_mag)
    w0 = gd.CartesianPhaseSpacePosition(pos=pos, vel=vel)

    # to get eccentricities, integrate orbits for 10 crossing times
    t_cross = gc_radius / v_mag

    dt = t_cross / 128.
    n_steps = 8192 # 64 crossing times

    ecc = np.zeros_like(t_cross)
    r_f = np.zeros_like(t_cross)
    for i in range(n_clusters):
        logger.debug('Integrating {}, dt={:.3f}'.format(i, dt[i]))
        if np.isnan(dt[i]):
            ecc[i] = np.nan
            r_f[i] = np.nan
            continue

        w = mw_potential.integrate_orbit(w0[i], dt=dt[i], n_steps=n_steps,
                                         Integrator=gi.DOPRI853Integrator)
        ecc[i] = w.eccentricity()
        r_f[i] = np.sqrt(np.sum(w.pos[:,-1]**2)).value

        if np.isnan(ecc[i]):
            # plt.figure()
            # w.plot()
            # plt.show()
            raise ValueError("Failed to compute eccentricity from "
                             "integrated orbit {} -- increase n_steps!"
                             .format(i))

    # plot the eccentricity distribution, initial radial profile, final radial profile
    idx = np.isfinite(ecc)
    if idx.sum() != len(idx):
        logger.warning("{}/{} failed eccentricities".format(n_clusters-idx.sum(),
                                                            n_clusters))

    fig,axes = plt.subplots(1, 2, figsize=(10,5))

    axes[0].hist(ecc[idx], bins=np.linspace(0,1,16))
    axes[0].set_xlabel('$e$')

    # compute radial profile of clusters -- initial and final
    bins = np.logspace(-1, 3, 32)
    H_i,_ = np.histogram(gc_radius, bins=bins)
    H_f,_ = np.histogram(r_f[idx], bins=bins)

    V = 4/3*np.pi*(bins[1:]**3 - bins[:-1]**3)
    bin_cen = (bins[1:]+bins[:-1])/2.
    axes[1].loglog(bin_cen, [gc_prob_density(x) for x in bin_cen],
                   marker=None, lw=2., ls='--', label='target')
    axes[1].loglog(bin_cen, H_i / V / n_clusters, marker=None, label='initial')
    axes[1].loglog(bin_cen, H_f / V / len(r_f), marker=None, label='final', color='r')

    axes[1].legend(loc='lower left')
    axes[1].set_xlabel('$r$')
    axes[1].set_ylabel('$n(r)$')

    fig.tight_layout()
    fig.savefig(join(paths.plots, 'ecc-radial-profile-{}.pdf').format(df_name))

    # Write out the initial conditions and cluster properties
    # TODO: for now, this is fine. but i might want to just write to the same ecsv file?
    with h5py.File(paths.gc_w0.format(df_name), 'w') as f:
        f.attrs['n'] = n_clusters

        d = f.create_dataset('w0_pos', data=w0.pos)
        d.attrs['unit'] = 'kpc'

        d = f.create_dataset('w0_vel', data=w0.vel)
        d.attrs['unit'] = 'kpc/Myr'

        d = f.create_dataset('mass', data=gc_mass)
        d.attrs['unit'] = 'Msun'

        f.create_dataset('ecc', data=ecc)

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
