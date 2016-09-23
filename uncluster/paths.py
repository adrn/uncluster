# Standard library
import os
from os.path import abspath, join, split, exists

__all__ = ['Paths']

class Paths(object):

    """
    A class that maintains various project paths for making plots and
    caching intermediate data products.

    Parameters
    ----------
    script__file__ : str
        Called from within a script in the ``scripts`` path, this should be
        the builtin ``__file__`` variable.

    """

    def __init__(self):
        self.root = abspath(join(abspath(os.getcwd()), ".."))

        # first, make sure we're in the scripts directory:
        if not exists(join(self.root, "scripts")):
            raise IOError("You must run this script from inside the scripts directory:\n{}"
                          .format(join(self.root, "scripts")))

        self.cache = join(self.root, "cache")
        self.plots = join(self.root, "plots")
        self.figures = join(self.root, "paper", "figures")

        for path in [self.cache, self.plots, self.figures]:
            os.makedirs(str(path), exist_ok=True)

        # store paths for special cache files
        self.gc_properties = join(self.cache, "1-gc-properties.ecsv")
        self.gc_w0 = join(self.cache, "2-w0-{}.hdf5")
