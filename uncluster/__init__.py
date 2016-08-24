from .gnedin_mass_radius import *

def get_output_path(f):
    import os
    from os.path import abspath, split, join, exists

    _root_path = abspath(join(split(abspath(f))[0], ".."))
    OUTPUT_PATH = join(_root_path, "output")
    if not exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    return OUTPUT_PATH
