"""Module implementing functions that could be calibrated to match observed
line luminosity functions."""


def logsfr_poly2_mod(logsfr, params):
    """Simple SFR modifier that could be calibrated to match observed line
    luminosity functions.

    Parameters
    ----------
    logsfr : ndarray, shape (n, )
        Log10 star formation rate in units of Msun/yr

    Returns
    -------
    mod_sfr : ndarray, shape (n, )
        Modified log10 star formation rate in units of Msun/yr
    """
    a, b, c, d = params
    mod_logsfr = b + c*(logsfr+a) + d*(logsfr+a)**2
    return mod_logsfr
