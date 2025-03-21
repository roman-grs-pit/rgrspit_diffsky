""" """

import numpy as np
import pytest

from .. import load_abacus as la

DEFAULT_HALOCAT_FN = ""
NO_ABACUS_MSG = "Must have access to AbacusSummit data to run this test"


@pytest.mark.skipif(not la.HAS_ABACUS_DATA, reason=NO_ABACUS_MSG)
def test_load_abacus_halo_catalog():
    halos = la.load_abacus_halo_catalog(DEFAULT_HALOCAT_FN)
    for colname, arr in halos.items():
        assert np.all(np.isfinite(arr))
