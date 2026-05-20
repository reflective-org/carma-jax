"""NCAR-test parity track.

Compares carma-jax (faithful JAX and diffrax paths) against the bench
outputs from the NCAR Fortran test suite at
``../original-carma/CARMA/tests/bench/``. Each test isolates one physics
mode (growth, coag, nuc, fall, sulfate variants…).
"""
