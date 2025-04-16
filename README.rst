rgrspit_diffsky
============
This repository uses the Diffsky model to make mocks for the Roman GRS PIT.

Documentation
-------------
See https://rgrspit-diffsky.readthedocs.io for documentation and code demos.


Installation
------------
To install rgrspit_diffsky into your environment from the source code::

    $ cd /path/to/root/rgrspit_diffsky
    $ pip install .

Testing
-------
To run the suite of unit tests::

    $ cd /path/to/root/rgrspit_diffsky
    $ pytest

To build html of test coverage::

    $ pytest -v --cov --cov-report html
    $ open htmlcov/index.html
