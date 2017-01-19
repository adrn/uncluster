# Standard library
from pathlib import Path

# Project
from ..paths import _find_root

def test_find_root():
    root_path = Path("/some/stupid/path/uncluster")

    test_path = _find_root(root_path)
    assert root_path == test_path

    test_path = Path("/some/stupid/path/uncluster/scripts")
    test_path = _find_root(test_path)
    assert root_path == test_path

    test_path = Path("/some/stupid/path/uncluster/uncluster")
    test_path = _find_root(test_path)
    assert root_path == test_path

    test_path = Path("/some/stupid/path/uncluster/paper/figures")
    test_path = _find_root(test_path)
    assert root_path == test_path
