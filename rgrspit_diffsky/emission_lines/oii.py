"""Module implementing scaling relations for OII emission."""

import numpy as np  # type: ignore
from jax import numpy as jnp
from jax import random as jran

K98_OII_COEF = 1.4e-41
OII_RESCALING = 1.78
OII_LOGNOISE = 0.11


def get_observed_OII(
    lum_oii, ran_key, *, rescaling_factor=OII_RESCALING, noise=OII_LOGNOISE
):
    """Calculate observed [OII] luminosity with empirical correction

    Parameters
    ----------
    lum_oii : array, shape (n, )

    ran_key : jax.random.seed()

    rescaling_factor : float

    noise : float
        Lognormal noise added to the rescaled line luminosity

    Returns
    -------
    lum_oii_obs : array, shape (n,)

    Notes
    -----
    Credit to Wei Liu for the calibration of the empirical correction.

    """
    loglum = jnp.log10(lum_oii * rescaling_factor)
    loglum_obs = jran.normal(ran_key, shape=lum_oii.shape) * noise + loglum
    lum_oii_obs = 10**loglum_obs
    return lum_oii_obs


# pylint: disable=invalid-name
def sfr_to_OII3727_K98(sfr):
    """Convert star formation rate to OII forbidden-line doublet luminosity
    with Kennicutt (1998) relation.

    SFR(OII) [M_sun yr^{-1}] = (1.4 ± 0.4) * 10^{-41} * L_{OII} [erg s^{-1}]

    Kennicutt, R. C., J.: 1998, ARAA 36, 189
    Gallagher, J. S., Hunter, D. A., and Bushouse, H.: 1989, AJ 97, 700
    Kennicutt, R. C., J.: 1992, ApJ, 388, 310

    Parameters
    ----------
    sfr : ndarray, shape (n, )
        Star formation rate in units of Msun/yr

    Returns
    -------
    l_oii : ndarray, shape (n, )
        OII luminosity at 3727 in units of erg/s

    """
    if np.any(sfr < 0):
        raise ValueError("Star formation rate cannot be negative")
    return sfr / K98_OII_COEF
