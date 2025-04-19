"""This module loads catalogs of halos in the AbacusSummit dataset"""

import os

DRN_NERSC = "/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit"

halo_array_columns = ['id', 'npart', 'mass', 'pos', 'vel', 'radius', 'concentration']
halo_header_columns = ['lbox']
halo_columns = halo_array_columns + halo_header_columns

try:
    assert os.path.isdir(DRN_NERSC)
    HAS_ABACUS_DATA = True
except AssertionError:
    HAS_ABACUS_DATA = False

try:
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    HAS_ABACUS_DEPS = True
except ImportError:
    HAS_ABACUS_DEPS = False


def load_abacus_halo_catalog(fname):
    if not HAS_ABACUS_DEPS:
        raise ImportError("abacusutils is required to read abacus data")

    halos = {}
    compaso_catalog = CompaSOHaloCatalog(fname,fields=['id','N','x_com','v_com','r100_com','r10_com'])

    # sim information
    halos['lbox'] = compaso_catalog.header['BoxSize']
    halos['id'] = compaso_catalog.halos['id']
    halos['npart'] = compaso_catalog.halos['N']
    halos['mass'] = compaso_catalog.halos['N'] * compaso_catalog.header['ParticleMassHMsun']
    halos['pos'] = compaso_catalog.halos['x_com']
    halos['vel'] = compaso_catalog.halos['v_com']
    halos['radius'] = compaso_catalog.halos['r100_com']
    halos['concentration'] = compaso_catalog.halos['r100_com'] / compaso_catalog.halos['r10_com']

    return halos
