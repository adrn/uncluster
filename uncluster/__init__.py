from ._astropy_init import *

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from . import config
    from . import cluster_distributions
    from . import cluster_massloss
    from .paths import Paths
    paths = Paths()
