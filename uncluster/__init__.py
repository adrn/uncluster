# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is an Astropy affiliated package.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

class OutputPaths(object):

    def __init__(self, script__file__):
        import os
        from os.path import abspath, join, split, exists
        from pathlib import Path

        _root_path = Path(abspath(join(split(abspath(script__file__))[0], "..")))

        self.cache = _root_path / "cache"
        self.plot = _root_path / "plots"

        for path in [self.cache, self.plot]:
            if not path.exists:
                os.makedirs(path)

        # names of cache filenames
        self.gc_properties = self.cache.joinpath("1-gc-properties.ecsv")

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from .conf import *
    from . import cluster_distributions
    from . import cluster_massloss
