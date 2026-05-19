"""Pack/unpack between the structured CARMA state and a flat diffrax y vector.

Scope (cut 1): single-column sulfate. We carry::

    pc[nbin, nelem]   particle concentrations per bin per element
    gc[ngas]          gas-phase mass concentrations (H2SO4, H2O)
    T                 temperature (scalar)

into a flat 1-D array of length `nbin*nelem + ngas + 1` for the diffrax
state vector ``y``. The vertical dimension (NZ=1) is squeezed out — there is
exactly one cell. The faithful `carma` package keeps NZ for general
multi-level columns; here we drop it because diffrax operates per-cell.

The pack/unpack helpers are pure jnp ops (no Python branches on dynamic data)
so they are safe inside ``@jax.jit``.
"""
from typing import NamedTuple, Tuple

import jax.numpy as jnp


class StateShape(NamedTuple):
    """Sizes used to slice a packed y vector. Static — passed via closure."""
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
    """Concatenate (pc, gc, T) into one flat array.

    Args:
        pc: shape (nbin, nelem) — particle concentrations [#/cm^3 of air]
        gc: shape (ngas,)       — gas mass concentrations  [g/cm^3 of air]
        T:  shape ()            — temperature [K]

    Returns:
        Flat 1-D array of length nbin*nelem + ngas + 1.
    """
    return jnp.concatenate([pc.reshape(-1), gc, jnp.atleast_1d(T)])


def unpack(y: jnp.ndarray,
           shape: StateShape) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Inverse of pack.

    Returns:
        pc : (nbin, nelem)
        gc : (ngas,)
        T  : ()    — scalar
    """
    pc_flat = y[: shape.n_pc]
    gc = y[shape.n_pc : shape.n_pc + shape.ngas]
    T = y[shape.n_pc + shape.ngas]   # scalar
    pc = pc_flat.reshape(shape.nbin, shape.nelem)
    return pc, gc, T
