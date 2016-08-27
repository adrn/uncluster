*********
Uncluster
*********

The broad overview of what happens here is:

1. Choose radial and mass distributions for the initial globular cluster
   population. The radial distribution / density profile must be a
   :class:`gala.potential.PotentialBase` subclass because we'll use it later to
   sample from a DF.
2. Generate samples from the mass distribution so that the total mass in
   globular clusters is some fraction of the total stellar mass of the Galaxy,
   generate (mean) orbital radii by sampling from the density profile (1.).
3. Choose a DF to use (spherical isotropic, radially biased, tangentially
   biased, or accreted with a few satellite galaxies) and generate velocities
   for each cluster to match the DF embedded in a given background potential.
4. Integrate the orbits of the clusters in the total background potential and
   determine eccentricities.
5. Solve for the mass-loss history of the cluster using the differential
   equations in Gnedin et al. (2014), including the eccentricities measured from
   the integrated cluster orbits.
6. Run streakline-like models along each cluster orbit and post-process the
   samples to impose the mass-loss histories from step 5.
