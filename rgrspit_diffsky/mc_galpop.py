"""Generate a Monte Carlo realization of the galaxy distribution
for an input catalog of AbacusSummit host halos"""

from diffmah.diffmahpop_kernels.bimod_censat_params import DEFAULT_DIFFMAHPOP_PARAMS
from diffmah.diffmahpop_kernels.mc_bimod_cens import mc_diffmah_params_singlecen
from diffmah.diffmahpop_kernels.mc_bimod_sats import mc_diffmah_params_singlesat
from diffmah.diffmahpop_kernels.param_utils import mc_select_diffmah_params
from diffsky.mass_functions.mc_subs import generate_subhalopop
from dsps.cosmology.flat_wcdm import _age_at_z_kern, age_at_z0
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

_POP = (None, 0, None, 0, None)
mc_diffmah_params_cenpop = jjit(vmap(mc_diffmah_params_singlecen, in_axes=_POP))
mc_diffmah_params_satpop = jjit(vmap(mc_diffmah_params_singlesat, in_axes=_POP))


def mc_diffmah_params_halopop_synthetic_subs(
    ran_key,
    mhost_at_z_obs,
    z_obs,
    lgmp_min,
    cosmo_params,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
):
    mah_params_cens = mc_diffmah_params_cens(
        ran_key,
        mhost_at_z_obs,
        z_obs,
        cosmo_params,
        diffmahpop_params=diffmahpop_params,
    )
    subs_mhalo_at_z_obs, subs_host_halo_indx = mc_subhalo_mass(
        ran_key, mhost_at_z_obs, lgmp_min
    )
    mah_params_sats = mc_diffmah_params_sats(
        ran_key,
        subs_mhalo_at_z_obs,
        z_obs,
        cosmo_params,
        diffmahpop_params=diffmahpop_params,
    )
    return mah_params_cens, mah_params_sats, subs_host_halo_indx, subs_mhalo_at_z_obs


def mc_subhalo_mass(ran_key, host_halo_mass, lgmp_min):
    subhalo_info = generate_subhalopop(ran_key, host_halo_mass, lgmp_min)
    subs_lgmu, subs_lgmhost, subs_host_halo_indx = subhalo_info
    subs_logmh_at_z = subs_lgmu + subs_lgmhost
    subhalo_mass = 10**subs_logmh_at_z
    return subhalo_mass, subs_host_halo_indx


def mc_diffmah_params_cens(
    ran_key,
    mhalo_at_z_obs,
    z_obs,
    cosmo_params,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
):
    n_halos = mhalo_at_z_obs.size
    params_key, mah_type_key = jran.split(ran_key, 2)
    ran_keys = jran.split(params_key, n_halos)
    lgmh_at_z_obs = jnp.log10(mhalo_at_z_obs)
    t0 = age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t0)
    t_obs = _age_at_z_kern(z_obs, *cosmo_params)
    _res = mc_diffmah_params_cenpop(
        diffmahpop_params, lgmh_at_z_obs, t_obs, ran_keys, lgt0
    )
    mah_params_early, mah_params_late, frac_early_cens = _res

    uran_early = jran.uniform(mah_type_key, minval=0, maxval=1, shape=(n_halos,))
    mc_is_early = uran_early < frac_early_cens
    mah_params = mc_select_diffmah_params(
        mah_params_early, mah_params_late, mc_is_early
    )
    return mah_params


def mc_diffmah_params_sats(
    ran_key,
    mhalo_at_z_obs,
    z_obs,
    cosmo_params,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
):
    n_halos = mhalo_at_z_obs.size
    params_key, mah_type_key = jran.split(ran_key, 2)
    ran_keys = jran.split(params_key, n_halos)
    lgmh_at_z_obs = jnp.log10(mhalo_at_z_obs)
    t0 = age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t0)
    t_obs = _age_at_z_kern(z_obs, *cosmo_params)
    _res = mc_diffmah_params_satpop(
        diffmahpop_params, lgmh_at_z_obs, t_obs, ran_keys, lgt0
    )
    mah_params_early, mah_params_late, frac_early_cens = _res

    uran_early = jran.uniform(mah_type_key, minval=0, maxval=1, shape=(n_halos,))
    mc_is_early = uran_early < frac_early_cens
    mah_params = mc_select_diffmah_params(
        mah_params_early, mah_params_late, mc_is_early
    )
    return mah_params
