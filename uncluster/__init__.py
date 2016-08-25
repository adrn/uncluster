# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is an Astropy affiliated package.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from .gnedin_mass_radius import *
    from .conf import *

    def get_output_path(filename):
        """
        Given the filename of a module in `scripts`, get the output path.
        """
        import os
        from os.path import abspath, split, join, exists

        _root_path = abspath(join(split(abspath(filename))[0], ".."))
        OUTPUT_PATH = join(_root_path, "output")
        if not exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)
        return OUTPUT_PATH
