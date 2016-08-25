"""
    Solve for the mass-loss rates and destruction times for each cluster given
    its eccentricity, initial mass, and initial radius.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from os.path import join, exists

# Third-party
from astropy import log as logger
from astropy.table import QTable
import astropy.units as u
import h5py
import numpy as np

from uncluster import get_output_path
from uncluster.conf import t_evolve
from uncluster.cluster_massloss import solve_mass_radius
OUTPUT_PATH = get_output_path(__file__)

def main(gc_props_filename, output_filename):
    if not exists(gc_props_filename):
        raise IOError("File '{}' does not exist -- have you run 1-make-cluster-props.py?"
                      .format(gc_props_filename))
    gc_props = QTable.read(gc_props_filename, format='ascii.ecsv')

    t_grid = np.linspace(0., t_evolve.to(u.Gyr).value, 4096)
    gc_mass = gc_props['mass'].to(u.Msun).value
    gc_radius = gc_props['radius'].to(u.kpc).value

    disrupt_idx, m, r = solve_mass_radius(gc_mass, gc_radius, t_grid)

    t_disrupt = t_grid[disrupt_idx+1]

    # set disruption time to NaN for those that don't disrupt
    t_disrupt[t_disrupt == t_evolve.to(u.Gyr).value] = np.nan

    logger.info("{}/{} clusters survived".format(np.isnan(t_disrupt).sum(), len(t_disrupt)))

    # write to hdf5 file
    with h5py.File(output_filename, 'w') as f:
        f.create_dataset('time', data=t_grid)

        cl = f.create_group('clusters')
        for i in range(len(gc_props)):
            g = cl.create_group(str(i))
            g.create_dataset('mass', data=m[i])
            g.create_dataset('radius', data=r[i])
            g.attrs['t_disrupt'] = t_disrupt[i]

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

    parser.add_argument("--gc-properties", dest="gc_props_filename", default=None,
                        type=int, help="Path to ECSV file containing the sampled "
                                       "cluster properties (i.e. mass and orbital radius)")
    parser.add_argument("--output", dest="output_filename", default=None,
                        type=int, help="Output path for the HDF5 file written by this script.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    gc_props_filename = args.gc_props_filename
    if gc_props_filename is None:
        gc_props_filename = join(OUTPUT_PATH, "1-gc-properties.ecsv")

    output_filename = args.output_filename
    if output_filename is None:
        output_filename = join(OUTPUT_PATH, "2-gc-evolution.hdf5")

    np.random.seed(args.seed)
    main(gc_props_filename, output_filename)
