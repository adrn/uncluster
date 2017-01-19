*********
Uncluster
*********

The broad overview of what happens here is:

1. Choose radial and mass distributions for the initial globular cluster
   population. (The radial distribution / density profile must be a
   :class:`~gala.potential.PotentialBase` subclass because we'll use it later to
   sample from a DF.) Generate samples from these distributions so that the
   total mass in globular clusters is some fraction of the total stellar mass of
   the Galaxy, generate (mean) orbital radii by sampling from the radial
   distribution.
2. Choose a DF to use (spherical isotropic, radially biased, tangentially
   biased, or accreted with a few satellite galaxies) and generate velocities
   for each cluster to match the DF embedded in a given background
   potential. Integrate the orbits of the clusters in the total background
   potential and determine eccentricities.
3. Solve for the mass-loss history of the cluster using the differential
   equations in Gnedin et al. (2014), including the eccentricities measured
   from the integrated cluster orbits. Run streakline-like models along each
   cluster orbit and post-process the samples to impose the solved mass-loss
   histories.
4. Paint stellar populations (and abundances???) on to the star particles in
   each stream. Generate mock images with a background...


Re-initializing the generated code
==================================

Some of the files that specify the parameters and evolution of the Milky Way
potential model were auto-generated and included in the repository. You
shouldn't need to re-generated these files, unless you'd like to re-do the
entire experiment with a new set of Milky Way mass measurements (data). To
re-generate these files, you have to run the ``setup_potential.py`` script. This
script generates the files:

   - ``uncluster/potential/potential_config.py``
   - ``uncluster/potential/src/cosmo_arrays.h``
   - ``data/apw_menc.txt``

These files contain (a) best-fit parameters for the halo and nucleus components
(we fix the disk and bulge models), and (b) C arrays that specify the time
evolution of the different potential parameters so we can interpolate to get the
potential at any given time.

Reproducing the results
=======================

Here I'll describe all of the steps needed to reproduce the results starting
from a clean clone of this repository.

1. ``setup_df.py`` to generate grid of E-DF values
2. ``make_clusters.py`` to generate initial conditions and orbits for clusters
3. ``make_streams.py`` to generate mock streams for the clusters

API
===

.. automodapi:: uncluster

.. automodapi:: uncluster.cluster_distributions.apw

.. automodapi:: uncluster.cluster_distributions.gnedin

.. automodapi:: uncluster.cluster_massloss
