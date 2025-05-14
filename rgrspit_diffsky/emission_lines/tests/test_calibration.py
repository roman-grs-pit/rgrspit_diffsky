"""Tests for emission line calibration functions."""

import numpy as np
from rgrspit_diffsky.emission_lines.calibration import logsfr_poly2_mod


def test_logsfr_poly2_mod_basic():
    """Test basic functionality of the polynomial SFR modifier."""
    # Test parameters that don't modify the SFR (a=0, b=0, c=1, d=0)
    params = (0.0, 0.0, 1.0, 0.0)
    logsfr = np.array([-1.0, 0.0, 1.0])
    modified = logsfr_poly2_mod(logsfr, params)
    assert np.allclose(modified, logsfr)


def test_logsfr_poly2_mod_offset():
    """Test SFR modifier with constant offset."""
    # Test parameters that add a constant offset (a=0, b=0.5, c=1, d=0)
    params = (0.0, 0.5, 1.0, 0.0)
    logsfr = np.array([-1.0, 0.0, 1.0])
    expected = logsfr + 0.5
    modified = logsfr_poly2_mod(logsfr, params)
    assert np.allclose(modified, expected)


def test_logsfr_poly2_mod_quadratic():
    """Test SFR modifier with quadratic term."""
    # Test parameters that add a quadratic term (a=0, b=0, c=1, d=0.1)
    params = (0.0, 0.0, 1.0, 0.1)
    logsfr = np.array([-1.0, 0.0, 1.0])
    expected = logsfr + 0.1 * logsfr**2
    modified = logsfr_poly2_mod(logsfr, params)
    assert np.allclose(modified, expected)


def test_logsfr_poly2_mod_physical_range():
    """Test SFR modifier over physically reasonable SFR range."""
    # Test over typical SFR range (-3 to 3 in log10)
    logsfr = np.linspace(-3, 3, 100)
    params = (0.0, 0.0, 1.0, 0.1)
    modified = logsfr_poly2_mod(logsfr, params)

    # Check that the modification preserves the order of SFRs
    assert np.all(np.diff(modified) > 0)

    # Check that the modification doesn't change the sign of the SFR
    assert np.all(np.sign(modified) == np.sign(logsfr))


def test_logsfr_poly2_mod_parameters():
    """Test SFR modifier with different parameter combinations."""
    logsfr = np.array([-1.0, 0.0, 1.0])

    # Test case 1: Linear scaling (a=0, b=0, c=2, d=0)
    params = (0.0, 0.0, 2.0, 0.0)
    expected = 2.0 * logsfr
    modified = logsfr_poly2_mod(logsfr, params)
    assert np.allclose(modified, expected)

    # Test case 2: Quadratic enhancement (a=0, b=0, c=1, d=0.2)
    params = (0.0, 0.0, 1.0, 0.2)
    expected = logsfr + 0.2 * logsfr**2
    modified = logsfr_poly2_mod(logsfr, params)
    assert np.allclose(modified, expected)


def test_logsfr_poly2_mod_edge_cases():
    """Test SFR modifier with edge cases."""
    # Test with zero SFR
    params = (0.0, 0.0, 1.0, 0.1)
    logsfr = np.array([0.0])
    modified = logsfr_poly2_mod(logsfr, params)
    assert modified[0] == 0.0

    # Test with very small SFR
    logsfr = np.array([-10.0])
    modified = logsfr_poly2_mod(logsfr, params)
    # For negative SFR, the quadratic term will make it more negative
    expected = -10.0 + 0.1 * (-10.0)**2
    assert np.isclose(modified[0], expected)

    # Test with very large SFR
    logsfr = np.array([10.0])
    modified = logsfr_poly2_mod(logsfr, params)
    # For positive SFR, the quadratic term will make it more positive
    expected = 10.0 + 0.1 * (10.0)**2
    assert np.isclose(modified[0], expected)
