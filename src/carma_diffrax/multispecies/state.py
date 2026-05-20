"""Pack/unpack between multi-group CARMA state and flat diffrax y vector.

Scope: NZ=1 single column, but ngroup ≥ 1 and ngas ≥ 1. State::

    pc[nbin, nelem]   particle conc per bin per element [#/cm^3]
    gc[ngas]          gas-phase mass conc [g/cm^3 × zmet]
    T                 temperature [K]

flattened to length ``nbin*nelem + ngas + 1``. Mirrors the sulfate cut's
state.py but with arbitrary nelem/ngas.
"""
from typing import NamedTuple, Tuple

import jax.numpy as jnp


class MultiSpeciesShape(NamedTuple):
    """Static shape for pack/unpack. Resolves at JIT trace time."""
    nbin: int
    nelem: int
    ngas: int

    @property
    def n_pc(self) -> int:
        return self.nbin * self.nelem

    @property
    def total(self) -> int:
        return self.n_pc + self.ngas + 1


def pack(pc: jnp.ndarray, gc: jnp.ndarray, T: jnp.ndarray) -> jnp.ndarray:
    """(pc, gc, T) → flat 1-D y.

    Args:
        pc: (nbin, nelem)
        gc: (ngas,)
        T:  ()
    """
    return jnp.concatenate([pc.reshape(-1), gc, jnp.atleast_1d(T)])


def unpack(y: jnp.ndarray,
           shape: MultiSpeciesShape,
           ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Inverse of pack.

    Returns:
        pc : (nbin, nelem)
        gc : (ngas,)
        T  : scalar
    """
    pc_flat = y[: shape.n_pc]
    gc = y[shape.n_pc : shape.n_pc + shape.ngas]
    T = y[shape.n_pc + shape.ngas]
    pc = pc_flat.reshape(shape.nbin, shape.nelem)
    return pc, gc, T
