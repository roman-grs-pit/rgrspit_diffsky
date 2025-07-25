"""Generate a Monte Carlo realization of the galaxy distribution
for an input catalog of AbacusSummit host halos"""

import numpy as np
from diffmah.diffmah_kernels import DiffmahParams, _log_mah_kern, mah_halopop
from diffmah.diffmahpop_kernels.bimod_censat_params import DEFAULT_DIFFMAHPOP_PARAMS
from diffmah.diffmahpop_kernels.mc_bimod_cens import mc_diffmah_params_singlecen
from diffmah.diffmahpop_kernels.mc_bimod_sats import mc_diffmah_params_singlesat
from diffmah.diffmahpop_kernels.param_utils import mc_select_diffmah_params
from diffsky.mass_functions.mc_subs import generate_subhalopop
from diffstar.defaults import T_TABLE_MIN
from diffstar.utils import cumulative_mstar_formed_galpop
from diffstarpop import mc_diffstarpop_tpeak_sepms_satfrac as mcdsp
from diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from diffstarpop.param_utils import mc_select_diffstar_params
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


def mc_galpop_synthetic_subs(
    ran_key,
    logmhost,
    halo_radius,
    halo_pos,
    halo_vel,
    z_obs,
    lgmp_min,
    cosmo_params,
    Lbox,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
):
    """Generate a Monte Carlo realizaton of galaxies populating the input halos

    Parameters
    ----------
    ran_key : jax.random.key

    logmhost : ndarray, shape (n_hosts, )
        log10 of halo mass in units of Msun

    halo_radius : ndarray, shape (n_hosts, )
        Halo radius in units of Mpc

    halo_pos : ndarray, shape (n_hosts, 3)
        Comoving halo position in units of Mpc

    halo_vel : ndarray, shape (n_hosts, 3)
        Halo velocity in units of km/s

    z_obs : float
        Redshift of the halo catalog

    lgmp_min : float
        log10 of halo mass cutoff in Msun

    cosmo_params : namedtuple
        Field names: ('Om0', 'w0', 'wa', 'h')

    Lbox : float
        Comoving size of the periodic box in Mpc

    Returns
    -------
    galcat : dict
        Dictionary storing MAHs, pos/vel, and SFH info about cens and sats

    """
    n_cens = logmhost.size
    mah_key, rhalo_key, axes_key, sfh_key = jran.split(ran_key, 4)
    _res = _mc_diffmah_params_halopop_synthetic_subs(
        mah_key,
        logmhost,
        z_obs,
        lgmp_min,
        cosmo_params,
        diffmahpop_params=diffmahpop_params,
    )
    mah_params_cens, mah_params_sats, subs_host_halo_indx, subs_logmh_at_z_obs = _res

    subs_rhost = halo_radius[subs_host_halo_indx]
    subs_logmhost = logmhost[subs_host_halo_indx]
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
    subs_host_pos = halo_pos[subs_host_halo_indx]
    subs_host_vel = halo_vel[subs_host_halo_indx]

    subs_pos = jnp.mod(subs_host_pos + subs_host_centric_pos, Lbox)
    subs_vel = subs_host_vel + subs_host_centric_vel

    pos = np.concatenate((halo_pos, subs_pos))
    vel = np.concatenate((halo_vel, subs_vel))

    # For every sub, get diffmah params of its host halo
    subs_host_diffmah = DiffmahParams(
        *[x[subs_host_halo_indx] for x in mah_params_cens]
    )

    t_obs = _age_at_z_kern(z_obs, *cosmo_params)
    t0 = age_at_z0(*cosmo_params)
    lgt0 = np.log10(t0)

    mah_params = DiffmahParams(
        *[np.concatenate((x, y)) for x, y in zip(mah_params_cens, mah_params_sats)]
    )
    logmp0 = np.array(_log_mah_kern(mah_params, t0, lgt0))
    lgmp_t_obs = np.array(_log_mah_kern(mah_params, t_obs, lgt0))

    host_mah_params = DiffmahParams(
        *[np.concatenate((x, y)) for x, y in zip(mah_params_cens, subs_host_diffmah)]
    )
    lgmhost_at_t_inf = _log_mah_kern(host_mah_params, mah_params.t_peak, lgt0)

    subs_lgmh_at_t_inf = _log_mah_kern(mah_params_sats, mah_params_sats.t_peak, lgt0)
    subs_lgmhost_at_t_inf = _log_mah_kern(
        subs_host_diffmah, mah_params_sats.t_peak, lgt0
    )
    subs_lgmu_t_inf = subs_lgmh_at_t_inf - subs_lgmhost_at_t_inf
    hosts_lgmu_t_inf = np.zeros(n_cens)
    lgmu_t_inf = np.concatenate((hosts_lgmu_t_inf, subs_lgmu_t_inf))

    upid = np.concatenate((np.zeros(n_cens).astype(int) - 1, subs_host_halo_indx))

    t_table = jnp.linspace(T_TABLE_MIN, t_obs, 50)
    log_mah_table = mah_halopop(mah_params, t_table, lgt0)[1]
    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        mah_params,
        logmp0,
        upid,
        lgmu_t_inf,
        lgmhost_at_t_inf,
        t_obs - mah_params.t_peak,
        sfh_key,
        t_table,
    )
    _sfh_res = mcdsp.mc_diffstar_sfh_galpop(*args)
    sfh_ms, sfh_q, frac_q, mc_is_q = _sfh_res[2:]
    sfh_table = jnp.where(mc_is_q.reshape((-1, 1)), sfh_q, sfh_ms)
    smh_table = cumulative_mstar_formed_galpop(t_table, sfh_table)
    diffstar_params_ms, diffstar_params_q = _sfh_res[0:2]
    sfh_params = mc_select_diffstar_params(
        diffstar_params_q, diffstar_params_ms, mc_is_q
    )

    lgsfr_at_t_obs = np.log10(sfh_table[:, -1])
    logsm_t_obs = np.log10(smh_table[:, -1])
    logssfr_t_obs = lgsfr_at_t_obs - logsm_t_obs

    galcat = dict()
    galcat["mah_params"] = mah_params
    galcat["sfh_params"] = sfh_params
    galcat["host_mah_params"] = host_mah_params
    galcat["upid"] = upid
    galcat["pos"] = pos
    galcat["vel"] = vel
    galcat["logmp0"] = logmp0
    galcat["logmp_t_obs"] = lgmp_t_obs
    galcat["logmu_t_inf"] = lgmu_t_inf
    galcat["logsm_t_obs"] = logsm_t_obs
    galcat["logssfr_t_obs"] = logssfr_t_obs
    galcat["t_table"] = t_table
    galcat["log_mah_table"] = log_mah_table
    galcat["sfh_table"] = sfh_table

    galcat["t0"] = t0
    galcat["z_obs"] = z_obs
    galcat["t_obs"] = t_obs

    return galcat


def _mc_diffmah_params_halopop_synthetic_subs(
    ran_key,
    logmhost_at_z_obs,
    z_obs,
    lgmp_min,
    cosmo_params,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
):
    mah_params_cens = _mc_diffmah_params_cens(
        ran_key,
        logmhost_at_z_obs,
        z_obs,
        cosmo_params,
        diffmahpop_params=diffmahpop_params,
    )
    subs_logmh_at_z_obs, subs_host_halo_indx = _mc_subhalo_mass(
        ran_key, logmhost_at_z_obs, lgmp_min
    )
    mah_params_sats = _mc_diffmah_params_sats(
        ran_key,
        subs_logmh_at_z_obs,
        z_obs,
        cosmo_params,
        diffmahpop_params=diffmahpop_params,
    )
    return mah_params_cens, mah_params_sats, subs_host_halo_indx, subs_logmh_at_z_obs


def _mc_subhalo_mass(ran_key, logmhost, lgmp_min):
    subhalo_info = generate_subhalopop(ran_key, logmhost, lgmp_min)
    subs_lgmu, subs_lgmhost, subs_host_halo_indx = subhalo_info
    subs_logmh_at_z = subs_lgmu + subs_lgmhost
    return subs_logmh_at_z, subs_host_halo_indx


def _mc_diffmah_params_cens(
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


def _mc_diffmah_params_sats(
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
