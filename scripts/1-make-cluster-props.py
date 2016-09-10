"""
    Generate masses, mean orbital radii, and eccentricities for a sample of globular clusters
    following Gnedin et al. (2014).
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from os.path import join, exists
import sys

# Third-party
from astropy import log as logger
from astropy.table import QTable
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from uncluster.paths import Paths
paths = Paths()

# from uncluster.cluster_distributions.gnedin import sample_radii, sample_masses
from uncluster.cluster_distributions.apw import sample_radii, sample_masses
from uncluster.config import f_gc, M_tot

def main(overwrite=False):

    if exists(paths.gc_properties) and not overwrite:
        logger.info("Mass and radius sampling already done. Use -o (--overwrite) "
                    "to redo")
        sys.exit(0)

    # =========================================================
    # Masses
    # =========================================================

    # - Sample masses until the total mass in GCs is equal to a fraction (f_gc) of the
    #   total mass in stars (M_tot):
    maxiter = 32000
    gc_mass = sample_masses(size=maxiter)
    for i in range(maxiter):
        _sum = gc_mass[:i+1].sum()
        if _sum >= f_gc*M_tot:
            break

    if i == (maxiter-1):
        raise ValueError("Reached maximum number of iterations when sampling masses.")

    gc_mass = gc_mass[:i+1]
    N_gc = len(gc_mass)
    logger.info("Sampled {} cluster masses (M_tot = {:.2e})".format(N_gc, gc_mass.sum()))

    # =========================================================
    # Mean orbital radii
    # =========================================================

    # only take radii out to ~virial radius
    gc_radius = sample_radii(r_max=250., size=N_gc)

    # =========================================================
    # Make plots
    # =========================================================

    # plot enclosed mass profile for the clusters
    _r = np.logspace(-2, 2, 512) * u.kpc
    menc = np.zeros(_r.size)
    for i,_rr in enumerate(_r):
        idx = gc_radius < _rr
        menc[i] = gc_mass[idx].sum().value

    plt.figure(figsize=(5,5))
    plt.loglog(_r, menc, marker=None)
    plt.xlim(1E-3, 1E2)
    plt.ylim(1E6, 1E9)
    plt.xlabel(r"$r$ [kpc]")
    plt.ylabel(r"$M_{\rm GC}(<r)$ [M$_\odot$]")
    plt.title("Compare to Fig. 3 in G14", fontsize=16)
    plt.savefig(join(paths.plots, 'gc-enclosed-mass.pdf'))

    # =========================================================
    # Cache the output
    # =========================================================
    tbl = QTable({'mass': gc_mass, 'radius': gc_radius})
    tbl.write(paths.gc_properties, format='ascii.ecsv')

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

    main(overwrite=args.overwrite)
