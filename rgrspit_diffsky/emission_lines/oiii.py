""""""

from jax import numpy as jnp
from jax import random as jran

OIII_RESCALING = 1.0
OIII_LOGNOISE = 0.19


def get_observed_OIII(
    lum, ran_key, *, rescaling_factor=OIII_RESCALING, noise=OIII_LOGNOISE
):
    """Calculate observed [OIII] luminosity with empirical correction

    Parameters
    ----------
    lum : array, shape (n, )

    ran_key : jax.random.seed()

    rescaling_factor : float

    noise : float
        Lognormal noise added to the rescaled line luminosity

    Returns
    -------
    lum_OIII_obs : array, shape (n,)

    """
    loglum = jnp.log10(lum * rescaling_factor)
    loglum_obs = jran.normal(ran_key, shape=lum.shape) * noise + loglum
    lum_obs = 10**loglum_obs
    return lum_obs
