"""
    Generate masses, mean orbital radii, and eccentricities for a sample of globular clusters
    following Gnedin et al. (2014).

    TODO: the eccentricity distribution should be customizable. Right now, I assume Gaussian
    for simplicity.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from os.path import join

# Third-party
from astropy import log as logger
from astropy.table import QTable
import astropy.units as u
import numpy as np

from uncluster import get_output_path, sample_radii, sample_masses
from uncluster.conf import f_gc, M_min, M_max, M_tot, r_max
OUTPUT_PATH = get_output_path(__file__)

def main():

    # Mass
    # - First I need to draw masses until the total mass in GCs is equal to
    #   a fraction of the total mass in stars:
    maxiter = 32000
    gc_mass = sample_masses(M_min=M_min, M_max=M_max, size=maxiter)
    for i in range(maxiter):
        _sum = gc_mass[:i+1].sum()
        logger.debug("{} - M_tot={:.2e}".format(i, _sum))
        if _sum >= f_gc*M_tot:
            break

    if i == (maxiter-1):
        raise ValueError("Reached maximum number of iterations when sampling masses.")

    gc_mass = gc_mass[:i+1]
    N_gc = len(gc_mass)
    logger.info("Sampled {} cluster masses (M_tot = {:.2e})".format(N_gc, gc_mass.sum()))

    # Mean orbital radius
    gc_radius = sample_radii(r_max=r_max, size=N_gc)

    # ------------------------------------------------------------------------
    # a test plot:
    # import matplotlib.pyplot as plt
    # _r = np.logspace(-2, 2, 512) * u.kpc
    # menc = np.zeros(_r.size)
    # for i,_rr in enumerate(_r):
    #     idx = gc_radius < _rr
    #     menc[i] = gc_mass[idx].sum().value

    # plt.figure(figsize=(6,6))
    # plt.loglog(_r, menc, marker=None)
    # plt.xlim(1E-3, 1E2)
    # plt.ylim(1E6, 1E9)
    # plt.title("Compare to Fig. 3 in G14", fontsize=16)
    # plt.show()
    # ------------------------------------------------------------------------

    # Eccentricity
    # HACK: for now, use a truncated normal...
    from scipy.stats import truncnorm
    mean, std = 0.6, 0.3 # mean, stddev of my hack eccentricity distribution
    _a, _b = 0., 1. # bounds for eccentricity
    a, b = (_a - mean) / std, (_b - mean) / std
    gc_ecc = truncnorm(a, b, loc=mean, scale=std).rvs(N_gc)

    tbl = QTable({'mass': gc_mass, 'radius': gc_radius, 'ecc': gc_ecc})
    tbl.write(join(OUTPUT_PATH, "1-gc-properties.ecsv"), format='ascii.ecsv')

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
    # parser.add_argument("-f", dest="field_id", default=None, required=True,
    #                     type=int, help="Field ID")
    # parser.add_argument("-p", dest="plot", action="store_true", default=False,
    #                     help="Plot or not")

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
