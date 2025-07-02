""" """

import argparse

from rgrspit_diffsky.data_loaders import load_fake_abacus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "machine", help="Machine name where script is run", choices=["lcrc", "poboy"]
    )

    subcat = load_fake_abacus.load_fake_abacus_halos(n_halos=200)
