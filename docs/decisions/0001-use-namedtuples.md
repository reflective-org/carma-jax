# ADR 0001: Use NamedTuples for State Representation

## Status

Accepted

## Context

CARMA-JAX needs data structures for two roles:
1. **CarmaConfig** — static model configuration (dimensions, process flags, precomputed tables)
2. **CarmaState** — per-column atmospheric state (all JAX arrays, evolves during timestep)

Both must be compatible with JAX's functional transformation system (jit, vmap, scan, grad). Options considered:

- **NamedTuples**: Automatic JAX pytrees (since JAX 0.4.1+), immutable, `._replace()` for updates
- **equinox.Module**: Better ergonomics (`eqx.tree_at`, `eqx.filter_jit`), but adds dependency
- **dataclasses**: Not automatic pytrees, need `jax.tree_util.register_pytree_class`
- **flax.struct**: Adds flax dependency, designed for neural network state

## Decision

Use **NamedTuples** for both CarmaConfig and CarmaState.

## Consequences

**Positive:**
- Zero extra dependencies (only JAX)
- Automatic JAX pytree registration
- Immutable by default, enforcing pure functional style
- Simple, transparent, well-understood Python idiom
- Migration to equinox.Module is mechanical if needed later

**Negative:**
- `._replace()` syntax is verbose for deep updates (`state._replace(pc=state.pc.at[iz].set(new_val))`)
- No built-in distinction between static and dynamic fields (must use `functools.partial` or `static_argnums`)
- Large NamedTuples with many fields can be unwieldy
- No runtime validation of field types

**Mitigations:**
- Helper functions for common update patterns
- CarmaConfig is always static (partial/static_argnums), CarmaState is always dynamic
- Group related fields into sub-NamedTuples (CoagConfig, GrowthConfig, etc.)
