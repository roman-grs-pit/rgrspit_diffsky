""" """

import numpy as np

from .. import load_fake_abacus as lfa


def test_load_fake_abacus_halos():
    halos = lfa.load_fake_abacus_halos()
    for x in halos:
        assert np.all(np.isfinite(x))
