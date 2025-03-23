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

from .fake_sats import vector_utilities as vectu
from .fake_sats.ellipsoidal_nfw_phase_space import mc_ellipsoidal_nfw
from .fake_sats.ellipsoidal_velocities import calculate_virial_velocity

_POP = (None, 0, None, 0, None)
mc_diffmah_params_cenpop = jjit(vmap(mc_diffmah_params_singlecen, in_axes=_POP))
mc_diffmah_params_satpop = jjit(vmap(mc_diffmah_params_singlesat, in_axes=_POP))


def mc_halopop_synthetic_subs_with_positions(
    ran_key,
    logmhost_at_z_obs,
    halo_radius_at_z_obs,
    z_obs,
    lgmp_min,
    cosmo_params,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
):
    mah_key, rhalo_key, axes_key = jran.split(ran_key, 3)
    _res = mc_diffmah_params_halopop_synthetic_subs(
        mah_key,
        logmhost_at_z_obs,
        z_obs,
        lgmp_min,
        cosmo_params,
        diffmahpop_params=diffmahpop_params,
    )
    mah_params_cens, mah_params_sats, subs_host_halo_indx, subs_logmh_at_z_obs = _res

    subs_rhost = halo_radius_at_z_obs[subs_host_halo_indx]
    subs_logmhost = logmhost_at_z_obs[subs_host_halo_indx]
    n_sats = subs_host_halo_indx.size
    ZZ = jnp.zeros(n_sats)
    conc = ZZ + 5.0
    subs_sigma = calculate_virial_velocity(10**subs_logmhost, subs_rhost)
    major_axes = jran.uniform(axes_key, minval=0, maxval=1, shape=(n_sats, 3))
    major_axes = vectu.normalized_vectors(major_axes)
    b_to_a = jnp.ones(n_sats)
    c_to_a = jnp.ones(n_sats)

    subs_host_centric_pos, subs_host_centric_vel = mc_ellipsoidal_nfw(
        rhalo_key, subs_rhost, conc, subs_sigma, major_axes, b_to_a, c_to_a
    )
    ret = (*_res, subs_host_centric_pos, subs_host_centric_vel)
    return ret


def mc_diffmah_params_halopop_synthetic_subs(
    ran_key,
    logmhost_at_z_obs,
    z_obs,
    lgmp_min,
    cosmo_params,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
):
    mah_params_cens = mc_diffmah_params_cens(
        ran_key,
        logmhost_at_z_obs,
        z_obs,
        cosmo_params,
        diffmahpop_params=diffmahpop_params,
    )
    subs_logmh_at_z_obs, subs_host_halo_indx = mc_subhalo_mass(
        ran_key, logmhost_at_z_obs, lgmp_min
    )
    mah_params_sats = mc_diffmah_params_sats(
        ran_key,
        subs_logmh_at_z_obs,
        z_obs,
        cosmo_params,
        diffmahpop_params=diffmahpop_params,
    )
    return mah_params_cens, mah_params_sats, subs_host_halo_indx, subs_logmh_at_z_obs


def mc_subhalo_mass(ran_key, logmhost, lgmp_min):
    subhalo_info = generate_subhalopop(ran_key, logmhost, lgmp_min)
    subs_lgmu, subs_lgmhost, subs_host_halo_indx = subhalo_info
    subs_logmh_at_z = subs_lgmu + subs_lgmhost
    return subs_logmh_at_z, subs_host_halo_indx


def mc_diffmah_params_cens(
    ran_key,
    lgmh_at_z_obs,
    z_obs,
    cosmo_params,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
):
    n_halos = lgmh_at_z_obs.size
    params_key, mah_type_key = jran.split(ran_key, 2)
    ran_keys = jran.split(params_key, n_halos)
    t0 = age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t0)
    t_obs = _age_at_z_kern(z_obs, *cosmo_params)
    t_peak = jnp.zeros(n_halos) + t0  # no early-peaking host halos
    _res = mc_diffmah_params_cenpop(
        diffmahpop_params, lgmh_at_z_obs, t_obs, ran_keys, lgt0, t_peak=t_peak
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
    lgmh_at_z_obs,
    z_obs,
    cosmo_params,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
):
    n_halos = lgmh_at_z_obs.size
    params_key, mah_type_key = jran.split(ran_key, 2)
    ran_keys = jran.split(params_key, n_halos)
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
