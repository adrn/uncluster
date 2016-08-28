"""
    TODO: explain
"""

__author__ = "adrn <adrn@princeton.edu>"

# Standard library
from multiprocessing import Pool

# Third-party
from astropy import log as logger
from astropy.table import QTable
import astropy.units as u
import emcee
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from uncluster import OutputPaths
paths = OutputPaths(__file__)
from uncluster.config import t_evolve, mw_potential
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
            _ = sampler.run_mcmc(p0, 128)
        except Warning:
            logger.error("Failed to MCMC cluster {}!".format(i))
            return np.nan

        # the randint is redundant, but just being safe...
        return sampler.chain[np.random.randint(self.n_walkers), -1, 0]

    def __call__(self, args):
        i,m,r = args
        return self.work(i, m, r)

def main(overwrite=False):
    if not paths.gc_properties.exists():
        raise IOError("File '{}' does not exist -- have you run 1-make-cluster-props.py?"
                      .format(paths.gc_properties))

    # read radii and masses from cached file
    gc_props = QTable.read(str(paths.gc_properties), format='ascii.ecsv')

    t_grid = np.linspace(0., t_evolve.to(u.Gyr).value, 4096)
    gc_mass = gc_props['mass'].to(u.Msun).value
    gc_radius = gc_props['radius'].to(u.kpc).value

    # TODO: choose DF class at command line??

    # filename to cache interpolation grid
    interp_grid_path = paths.cache/"interp_grid_{}.ecsv".format(SphericalIsotropicDF.__name__)
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
    fig.savefig(str(paths.plot/'df-vs-energy.pdf'))

    # now I need to draw from the velocity distribution -- using emcee to do the sampling
    worker = Worker(df=iso, n_walkers=16)
    tasks = list(zip(range(len(gc_mass)), gc_mass, gc_radius))

    # HACK: only do 256 for now for speed
    DERP = 256
    gc_radius = gc_radius[:DERP]
    with Pool() as p: # use all CPUs
        v_mag = p.map(worker, tasks[:DERP])
    v_mag = np.array(v_mag)

    # need to turn the radius and velocity magnitude into 3D intial conditions
    pos,vel = iso.r_v_to_3d(gc_radius, v_mag)
    w0 = gd.CartesianPhaseSpacePosition(pos=pos, vel=vel)

    t_cross = gc_radius / v_mag
    ecc = np.zeros_like(t_cross)
    r_f = np.zeros_like(t_cross)

    for i in range(len(t_cross)):
        w = mw_potential.integrate_orbit(w0[i], dt=t_cross[i]/100., n_steps=2000)
        ecc[i] = w.eccentricity()
        r_f[i] = np.sqrt(np.sum(w.pos[:,-1]**2)).value

    plt.figure()
    plt.hist(ecc[np.isfinite(ecc)])

    plt.figure()
    bins = np.logspace(-1, 3, 32)
    H,_ = np.histogram(r_f, bins=bins)

    V = 4/3*np.pi*(bins[1:]**3 - bins[:-1]**3)
    bin_cen = (bins[1:]+bins[:-1])/2.

    plt.plot(bin_cen, [gc_prob_density(x) for x in bin_cen], marker=None, lw=2., ls='--')
    plt.loglog(bin_cen, H/V/r_f.size, marker=None)

    plt.xlabel('$r$')
    plt.ylabel('$n(r)$')

    plt.show()

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
