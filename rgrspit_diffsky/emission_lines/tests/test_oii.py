"""Tests for OII emission line calculations."""

import numpy as np
import pytest
from rgrspit_diffsky.emission_lines.oii import sfr_to_OII3727_K98, K98_OII_COEF


def test_oii_conversion_known_values():
    """Test OII conversion with known values from literature."""
    # Test case 1: SFR = 1 M_sun/yr should give L_OII = 7.14e40 erg/s
    sfr = np.array([1.0])
    expected_l_oii = 1.0 / K98_OII_COEF
    assert np.isclose(sfr_to_OII3727_K98(sfr)[0], expected_l_oii)

    # Test case 2: SFR = 10 M_sun/yr should give L_OII = 7.14e41 erg/s
    sfr = np.array([10.0])
    expected_l_oii = 10.0 / K98_OII_COEF
    assert np.isclose(sfr_to_OII3727_K98(sfr)[0], expected_l_oii)


def test_oii_conversion_array():
    """Test OII conversion with array inputs."""
    sfr = np.array([1.0, 10.0, 100.0])
    expected_l_oii = sfr / K98_OII_COEF
    assert np.allclose(sfr_to_OII3727_K98(sfr), expected_l_oii)


def test_oii_conversion_physical_range():
    """Test OII conversion for physically reasonable SFR range."""
    # Test SFR range from 0.1 to 1000 M_sun/yr
    sfr = np.logspace(-1, 3, 100)
    l_oii = sfr_to_OII3727_K98(sfr)

    # Check that all luminosities are positive
    assert np.all(l_oii > 0)

    # Check that the ratio between SFR and L_OII is constant
    ratio = sfr / l_oii
    assert np.allclose(ratio, K98_OII_COEF)


def test_oii_conversion_zero_sfr():
    """Test OII conversion with zero SFR."""
    sfr = np.array([0.0])
    l_oii = sfr_to_OII3727_K98(sfr)
    assert l_oii[0] == 0.0


def test_oii_conversion_negative_sfr():
    """Test OII conversion with negative SFR (should raise error)."""
    sfr = np.array([-1.0])
    with pytest.raises(ValueError):
        sfr_to_OII3727_K98(sfr)
