""" """

import argparse
import os
from glob import glob

from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import random as jran

from rgrspit_diffsky import mc_galpop
from rgrspit_diffsky.data_loaders import load_fake_abacus

NERSC_ROOT_DRN = ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "machine", help="Machine name where script is run", choices=["nersc", "poboy"]
    )

    parser.add_argument("drnout", help="Output directory")

    args = parser.parse_args()
    machine = args.machine
    drnout = args.drnout

    if machine == "nersc":
        data_drn = NERSC_ROOT_DRN
    elif machine == "poboy":
        raise NotImplementedError()

    fn_list = glob(os.path.join(data_drn, "file_pattern*.hdf5"))

    ran_key = jran.key(0)

    for fn in fn_list:
        ran_key, fn_key = jran.split(ran_key, 2)
        subcat = load_fake_abacus.load_fake_abacus_halos(n_halos=200)

        galcat = mc_galpop.mc_galpop_synthetic_subs(
            fn_key,
            logmhost,
            halo_radius,
            halo_pos,
            halo_vel,
            z_obs,
            lgmp_min,
            DEFAULT_COSMOLOGY,
            Lbox,
        )

        bn = os.path.basename(fn)
        fn_out = os.path.join(drnout, bn)

        # write galcat to disk
