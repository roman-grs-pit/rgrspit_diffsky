""" """

import numpy as np
import pytest

from .. import load_abacus as la

DEFAULT_HALOCAT_DIR = "/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/small/AbacusSummit_small_c000_ph3000/halos/z1.100"
NO_ABACUS_DATA_MSG = "Must have access to AbacusSummit data to run this test"
NO_ABACUS_DEPS_MSG = "Must have abacusnbody installed to run this test"


@pytest.mark.skipif(not la.HAS_ABACUS_DEPS, reason=NO_ABACUS_DEPS_MSG)
@pytest.mark.skipif(not la.HAS_ABACUS_DATA, reason=NO_ABACUS_DATA_MSG)
def test_load_abacus_halo_catalog():
    catalog = la.load_abacus_halo_catalog(DEFAULT_HALOCAT_DIR)
    for colname in la.halo_columns:
        assert colname in catalog.keys()
    for colname, val in catalog.items():
        assert colname in la.halo_columns
        assert np.all(np.isfinite(val))
