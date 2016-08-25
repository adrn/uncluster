"""
    Given masses, mass-loss rates, orbital radii, eccentricities, and destruction times for an
    initial population of clusters computed from reproducing the simulation in Gnedin et al.
    (2014), generate Streakline+scatter+full-disruption simulations for each globular cluster
    stream.

    This code makes the following simplifying assumptions:
        - Dynamical friction is neglected: we are interested in clusters with r > 5 kpc, where,
          even for the most massive globular clusters, dynamical friction has a negligible effect.
        -

    TODO:
    - In next script, take output from this file and paint stellar population, abundances on to
      star particles in each "stream"

"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from os.path import join, exists

# Third-party
from astropy import log as logger
import astropy.units as u
import h5py
import numpy as np
from scipy.integrate import odeint

import gala.coordinates as gc
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
from gala.units import galactic

from uncluster import get_output_path
from uncluster.conf import t_evolve
from uncluster.potential import potential as pot
OUTPUT_PATH = get_output_path(__file__)

def main(gc_evo_filename, output_filename):
    if not exists(gc_evo_filename):
        raise IOError("File '{}' does not exist -- have you run 2-evolve-clusters.py?"
                      .format(gc_evo_filename))

    with h5py.File(gc_evo_filename) as f:
        for i in range(f.attrs['n']):
            g = f['clusters/{}'.format(i)]

            t = f['time'][:g.attrs['disrupt_idx']+1]
            m = g['mass'][:]
            r = np.mean(g['radius'][:])
            ecc = g.attrs['ecc']
            break


    pot

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

    parser.add_argument("--gc-evolution", dest="gc_evo_filename", default=None,
                        type=int, help="Path to HDF5 file containing the mass and radius "
                                       "evolution for each cluster.")
    # parser.add_argument("--output", dest="output_filename", default=None,
    #                     type=int, help="Output path for the HDF5 file written by this script.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    gc_evo_filename = args.gc_evo_filename
    if gc_evo_filename is None:
        gc_evo_filename = join(OUTPUT_PATH, "2-gc-evolution.hdf5")

    # output_filename = args.output_filename
    # if output_filename is None:
    #     output_filename = join(OUTPUT_PATH, "2-gc-evolution.hdf5")

    np.random.seed(args.seed)
    main(gc_evo_filename, None)
