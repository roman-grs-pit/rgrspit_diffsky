"""Module implementing scaling relations for OII emission."""

import numpy as np # type: ignore

K98_OII_COEF = 1.4e-41

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
    return sfr - np.log10(K98_OII_COEF)
