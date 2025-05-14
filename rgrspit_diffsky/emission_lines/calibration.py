"""Module implementing functions that could be calibrated to match observed
line luminosity functions."""


def logsfr_poly2_mod(logsfr, params):
    """Simple SFR modifier that could be calibrated to match observed line
    luminosity functions.

    Parameters
    ----------
    logsfr : ndarray, shape (n, )
        Log10 star formation rate in units of Msun/yr
    params : tuple
        Polynomial coefficients (offset, constant, linear, quadratic)

    Returns
    -------
    mod_sfr : ndarray, shape (n, )
        Modified log10 star formation rate in units of Msun/yr
    """
    offset, constant, linear, quadratic = params
    mod_logsfr = constant + linear*(logsfr+offset) + quadratic*(logsfr+offset)**2
    return mod_logsfr
