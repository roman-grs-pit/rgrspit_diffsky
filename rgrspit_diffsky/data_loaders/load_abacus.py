"""This module loads catalogs of halos in the AbacusSummit dataset"""

import os

DRN_NERSC = "nersc/path/to/abacus/summit/halos"

try:
    assert os.path.isdir(DRN_NERSC)
    HAS_ABACUS_DATA = True
except AssertionError:
    HAS_ABACUS_DATA = False


def load_abacus_halo_catalog(fname):
    raise NotImplementedError()
