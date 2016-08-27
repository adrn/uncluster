"""
    Generate masses, mean orbital radii, and eccentricities for a sample of globular clusters
    following Gnedin et al. (2014).

    TODO: the eccentricity distribution should be customizable. Right now, I assume Gaussian
    for simplicity.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy import log as logger
from astropy.table import QTable
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from uncluster import OutputPaths
paths = OutputPaths(__file__)

# from uncluster.cluster_distributions.gnedin import sample_radii, sample_masses
from uncluster.cluster_distributions.apw import sample_radii, sample_masses
from uncluster.config import f_gc, M_tot

def main():
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

    gc_radius = sample_radii(size=N_gc)

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
    plt.savefig(str(paths.plot/'gc-enclosed-mass.pdf'))

    # =========================================================
    # Cache the output
    # =========================================================
    tbl = QTable({'mass': gc_mass, 'radius': gc_radius})
    tbl.write(str(paths.gc_properties), format='ascii.ecsv')

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

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    np.random.seed(args.seed)
    main()
