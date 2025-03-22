"""Generate a Monte Carlo realization of the galaxy distribution
for an input catalog of AbacusSummit host halos"""

from diffsky.mass_functions.mc_subs import generate_subhalopop
from jax import random as jran


def mc_subhalo_mass(ran_key, host_halo_mass, lgmp_min):
    subhalo_info = generate_subhalopop(ran_key, host_halo_mass, lgmp_min)
    subs_lgmu, subs_lgmhost, subs_host_halo_indx = subhalo_info
    subs_logmh_at_z = subs_lgmu + subs_lgmhost
    subhalo_mass = 10**subs_logmh_at_z
    return subhalo_mass, subs_host_halo_indx
