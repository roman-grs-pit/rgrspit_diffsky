"""This module loads Abacus-like synthetic halos for unit-testing purposes"""

import numpy as np
from diffsky.mass_functions.mc_diffmah_tpeak import mc_subhalos
from jax import random as jran


def load_fake_abacus_halos(n_halos=200):
    """Load a catalog of synthetic halos with the same attributes as Abacus halos.

    Notes
    -----
    This function is currently just a placeholder wrapper around the Diffsky
    MC generator, which returns a catalog of both host halos and subhalos.
    We will want to adapt this function so that the columns of the synthetic halos
    have the same names as the actual halo catalogs in AbacusSummit.

    """

    ran_key = jran.key(0)
    z_obs = 0.5
    hosts_logmh_at_z = np.linspace(11, 15, n_halos)
    lgmp_min = hosts_logmh_at_z.min()

    subcat = mc_subhalos(ran_key, z_obs, lgmp_min, hosts_logmh_at_z=hosts_logmh_at_z)

    return subcat
