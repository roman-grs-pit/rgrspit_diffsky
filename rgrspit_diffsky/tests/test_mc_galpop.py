""" """

import numpy as np
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from jax import random as jran

from .. import mc_galpop


def test_mc_subhalo_mass():
    """Enforce that mc_subhalo_mass generates subs with 10**lgmp_min<=Msub<=Mhost"""
    ran_key = jran.key(0)
    lgmp_min = 11.0
    n_halos = 2_500
    host_halo_mass = np.logspace(lgmp_min, 15, n_halos)
    subhalo_mass, subs_host_halo_indx = mc_galpop.mc_subhalo_mass(
        ran_key, host_halo_mass, lgmp_min
    )
    assert np.all(np.isfinite(subhalo_mass))
    assert np.all(subhalo_mass >= 10**lgmp_min)
    assert np.all(subs_host_halo_indx >= 0)
    assert np.all(subs_host_halo_indx < n_halos)

    subhalo_mhost = host_halo_mass[subs_host_halo_indx]
    assert np.all(subhalo_mhost >= subhalo_mass)


def test_mc_diffmah_params_cens():
    ran_key = jran.key(0)
    lgmp_min = 11.0
    n_halos = 2_500
    mhalo_at_z_obs = np.logspace(lgmp_min, 15, n_halos)
    z_obs = 0.5
    mah_params = mc_galpop.mc_diffmah_params_cens(
        ran_key, mhalo_at_z_obs, z_obs, DEFAULT_COSMOLOGY
    )
    assert np.all(np.isfinite(mah_params))


def test_mc_diffmah_params_sats():
    ran_key = jran.key(0)
    lgmp_min = 11.0
    n_halos = 2_500
    mhalo_at_z_obs = np.logspace(lgmp_min, 15, n_halos)
    z_obs = 0.5
    mah_params = mc_galpop.mc_diffmah_params_sats(
        ran_key, mhalo_at_z_obs, z_obs, DEFAULT_COSMOLOGY
    )
    assert np.all(np.isfinite(mah_params))
