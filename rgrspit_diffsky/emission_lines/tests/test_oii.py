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


def test_oii_luminosity_order_of_magnitude():
    """Test that OII luminosities are in physically reasonable ranges.

    For typical star-forming galaxies:
    - SFR ~ 1 M_sun/yr should give L_OII ~ 10^{40}-10^{41} erg/s
    - SFR ~ 10 M_sun/yr should give L_OII ~ 10^{41}-10^{42} erg/s
    - SFR ~ 100 M_sun/yr should give L_OII ~ 10^{42}-10^{43} erg/s

    References:
    - Kennicutt (1998) ARAA 36, 189
    - Kewley et al. (2004) AJ 127, 2002
    """
    # Test for a range of typical SFRs
    test_sfrs = np.array([1.0, 10.0, 100.0])
    l_oii = sfr_to_OII3727_K98(test_sfrs)

    # Expected ranges (in log10)
    min_expected = np.array([40.0, 41.0, 42.0])
    max_expected = np.array([41.0, 42.0, 43.0])

    log_l_oii = np.log10(l_oii)

    # Check each SFR case is within expected range
    for i in range(len(test_sfrs)):
        assert min_expected[i] <= log_l_oii[i] <= max_expected[i], (
            f"OII luminosity for SFR={test_sfrs[i]} M_sun/yr is "
            f"{log_l_oii[i]}, expected range: "
            f"[{min_expected[i]}, {max_expected[i]}]"
        )
