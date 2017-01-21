# Standard library
from pathlib import Path

__all__ = ['Paths']

def _find_root(path):
    packagename = 'uncluster'
    dirs = path.parts

    if packagename not in dirs:
        raise IOError("Can't find root {} path.".format(packagename))

    if dirs[-1] == packagename and dirs[-2] != packagename:
        # already in root path:
        return path

    else:
        i = dirs.index(packagename)
        return Path(*dirs[:i+1])

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
        self.root = _find_root(Path.cwd().absolute())

        self.cache = self.root / "cache"
        self.data = self.root / "uncluster/data"
        self.plots = self.root / "plots"
        self.figures = self.root / "paper/figures"

        for path in [self.cache, self.data, self.plots, self.figures]:
            path.mkdir(exist_ok=True)

        # store paths for special cache files
        self.gc_properties = self.cache / "1-gc-properties.ecsv"
        self.gc_w0 = self.cache / "2-w0-{}.hdf5"
