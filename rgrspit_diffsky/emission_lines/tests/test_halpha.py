"""Tests for H-alpha emission line calculations."""

import numpy as np
import pytest
from rgrspit_diffsky.emission_lines.halpha import (
    sfr_to_Halpha_KTC94,
    K98_HALPHA_COEF,
)
from rgrspit_diffsky.emission_lines.oii import sfr_to_OII3727_K98


def test_halpha_conversion_known_values():
    """Test H-alpha conversion with known values from literature."""
    # Test case 1: SFR = 1 M_sun/yr should give L_Halpha = 1.27e41 erg/s
    sfr = np.array([1.0])
    expected_l_halpha = 1.0 / K98_HALPHA_COEF
    assert np.isclose(sfr_to_Halpha_KTC94(sfr)[0], expected_l_halpha)

    # Test case 2: SFR = 10 M_sun/yr should give L_Halpha = 1.27e42 erg/s
    sfr = np.array([10.0])
    expected_l_halpha = 10.0 / K98_HALPHA_COEF
    assert np.isclose(sfr_to_Halpha_KTC94(sfr)[0], expected_l_halpha)


def test_halpha_conversion_array():
    """Test H-alpha conversion with array inputs."""
    sfr = np.array([1.0, 10.0, 100.0])
    expected_l_halpha = sfr / K98_HALPHA_COEF
    assert np.allclose(sfr_to_Halpha_KTC94(sfr), expected_l_halpha)


def test_halpha_conversion_physical_range():
    """Test H-alpha conversion for physically reasonable SFR range."""
    # Test SFR range from 0.1 to 1000 M_sun/yr
    sfr = np.logspace(-1, 3, 100)
    l_halpha = sfr_to_Halpha_KTC94(sfr)

    # Check that all luminosities are positive
    assert np.all(l_halpha > 0)

    # Check that the ratio between SFR and L_Halpha is constant
    ratio = sfr / l_halpha
    assert np.allclose(ratio, K98_HALPHA_COEF)


def test_halpha_conversion_zero_sfr():
    """Test H-alpha conversion with zero SFR."""
    sfr = np.array([0.0])
    l_halpha = sfr_to_Halpha_KTC94(sfr)
    assert l_halpha[0] == 0.0


def test_halpha_conversion_negative_sfr():
    """Test H-alpha conversion with negative SFR (should raise error)."""
    sfr = np.array([-1.0])
    with pytest.raises(ValueError):
        sfr_to_Halpha_KTC94(sfr)


def test_halpha_oii_ratio():
    """Test that H-alpha to OII ratio is consistent with literature values."""
    sfr = np.array([1.0])
    l_halpha = sfr_to_Halpha_KTC94(sfr)[0]
    l_oii = sfr_to_OII3727_K98(sfr)[0]

    # H-alpha/OII ratio should be approximately 1.8 (Kennicutt 1998)
    ratio = l_halpha / l_oii
    assert np.isclose(ratio, 1.8, rtol=0.1)


def test_halpha_luminosity_order_of_magnitude():
    """Test that H-alpha luminosities are in physically reasonable ranges.

    For typical star-forming galaxies:
    - SFR ~ 1 M_sun/yr should give L_Halpha ~ 10^{41}-10^{42} erg/s
    - SFR ~ 10 M_sun/yr should give L_Halpha ~ 10^{42}-10^{43} erg/s
    - SFR ~ 100 M_sun/yr should give L_Halpha ~ 10^{43}-10^{44} erg/s

    References:
    - Kennicutt (1998) ARAA 36, 189
    - Kennicutt et al. (1994) ApJ 435, 22
    - Moustakas et al. (2006) ApJ 642, 775
    """
    # Test for a range of typical SFRs
    test_sfrs = np.array([1.0, 10.0, 100.0])
    l_halpha = sfr_to_Halpha_KTC94(test_sfrs)

    # Expected ranges (in log10)
    min_expected = np.array([41.0, 42.0, 43.0])
    max_expected = np.array([42.0, 43.0, 44.0])

    log_l_halpha = np.log10(l_halpha)

    # Check each SFR case is within expected range
    for i in range(len(test_sfrs)):
        assert min_expected[i] <= log_l_halpha[i] <= max_expected[i], (
            f"H-alpha luminosity for SFR={test_sfrs[i]} M_sun/yr is "
            f"{log_l_halpha[i]}, expected range: "
            f"[{min_expected[i]}, {max_expected[i]}]"
        )
