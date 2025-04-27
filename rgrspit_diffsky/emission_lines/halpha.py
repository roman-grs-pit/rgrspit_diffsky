"""Module implementing scaling relations for Halpha emission."""
import numpy as np  # type: ignore

K98_HALPHA_COEF = 7.9e-42


def sfr_to_Halpha_KTC94(sfr):
    """Convert star formation rate to Halpha luminosity with Kennicutt,
    Tamblyn and Congdon (1994) relation as also described in Kennicutt (1989).

    SFR(Halpha) [M_sun yr^{-1}] = 7.9 * 10^{-42} * L_{Halpha} [erg s^{-1}]

    Kennicutt, R. C., J.: 1998, ARAA 36, 189
    Kennicutt, R. C., J., Tamblyn, P., and Congdon, C. E.: 1994, ApJ 435, 22

    Parameters
    ----------
    sfr : ndarray, shape (n, )
        Star formation rate in units of Msun/yr

    Returns
    -------
    l_halpha : ndarray, shape (n, )
        Halpha luminosity in units of erg/s
    """
    return sfr / K98_HALPHA_COEF
