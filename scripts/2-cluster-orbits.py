"""
    TODO: explain
"""

__author__ = "adrn <adrn@princeton.edu>"

# Third-party
from astropy import log as logger
from astropy.table import QTable
import astropy.units as u
import gala.potential as gp
from gala.units import galactic
import h5py
import matplotlib.pyplot as plt
import numpy as np

from uncluster import OutputPaths
paths = OutputPaths(__file__)
from uncluster.config import t_evolve, mw_potential
from uncluster.cluster_distributions.apw import gc_prob_density
from uncluster.distribution_function import SphericalIsotropicDF

def main(overwrite=False):
    if not paths.gc_properties.exists():
        raise IOError("File '{}' does not exist -- have you run 1-make-cluster-props.py?"
                      .format(paths.gc_properties))

    # read radii and masses from cached file
    gc_props = QTable.read(str(paths.gc_properties), format='ascii.ecsv')

    t_grid = np.linspace(0., t_evolve.to(u.Gyr).value, 4096)
    gc_mass = gc_props['mass'].to(u.Msun).value
    gc_radius = gc_props['radius'].to(u.kpc).value


    # filename to cache interpolation grid
    interp_grid_path = paths.cache/"interp_grid_{}.ecsv".format(SphericalIsotropicDF.__name__)
    if not interp_grid_path.exists() or overwrite:
        # generate a grid of energies to evaluate the DF on
        n_grid = 3
        r = np.array([[1E-4,0,0],
                      [1E3,0,0]]).T * u.kpc
        v = mw_potential.circular_velocity(r)
        E_min = (mw_potential.value(r[:,0]) + 0.5*v**2).decompose(galactic).value[0]
        E_max = (mw_potential.value(r[:,1]) + 0.5*v**2).decompose(galactic).value[0]

        iso = SphericalIsotropicDF(tracer=gc_prob_density,
                                   background=mw_potential,
                                   energy_grid=np.linspace(E_min, E_max, n_grid))

        tbl = QTable({'energy': iso._energy_grid * galactic['energy'],
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

    # for df in [isotropic]:
    plt.semilogy(-iso._energy_grid, iso._df_grid)
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
