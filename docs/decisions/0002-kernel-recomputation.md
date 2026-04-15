# ADR 0002: Coagulation Kernel Recomputation Strategy

## Status

Deferred to Phase 5 (orchestration)

## Context

The coagulation kernel `ckernel(NZ, NBIN, NBIN, NGROUP, NGROUP)` depends on atmospheric state (T, p, rmu) and particle properties (r_wet, bpm, vf, re). In the full model, these change every timestep due to:
- Latent heating (tsolve) changing T
- Condensational growth changing r_wet
- Vertical transport changing T, p, rhoa at each level

Fortran CARMA recomputes the kernel every timestep inside `CARMASTATE_Step`. Our Phase 1-2 validation computes it once and reuses — valid only for constant environmental conditions.

## Architecture Change Required

Current time loop (kernel outside scan):
```python
ckernel = setup_ckern(...)  # computed once
def step(pc, _):
    return microslow(pc, ckernel, ...), None
jax.lax.scan(step, pc, None, length=N)
```

Required time loop (kernel inside scan):
```python
def step(state, _):
    t, rhoa, rmu, r_wet, rhop_wet, pc = state
    vf, re, bpm = setup_vf_jit(t, rhoa, ...)
    ckernel = setup_ckern_jit(t, rhoa, ..., vf, bpm)
    pc = microslow_fn(pc, pcl, ckernel, ...)
    t = tsolve(...)      # update T from latent heat
    r_wet = grow(...)     # update radii from condensation
    return (t, rhoa, rmu, r_wet, rhop_wet, pc), outputs
jax.lax.scan(step, init_state, None, length=N)
```

Key changes:
1. **Scan carry becomes a tuple/NamedTuple** bundling all mutable state (T, p, rhoa, rmu, r_wet, rhop_wet, pc, gc) — not just pc
2. **setup_vf_jit and setup_ckern_jit called inside the scan body** — already JIT-compatible (no config dependency, pure array inputs)
3. **microslow_fn already takes ckernel as dynamic input** — no change needed
4. **XLA compiles a larger graph** — setup_vf + setup_ckern + microslow fused into one scan body. May increase compilation time but improves runtime by avoiding Python overhead

## Cost Analysis

Per-step overhead for kernel recomputation (47 bins, NZ=1):
- `setup_vf_jit`: 0.006 ms
- `setup_ckern_jit`: 0.040 ms
- `microslow_fn`: ~0.012 ms
- **Total per step**: ~0.06 ms
- **720 steps (12h at dt=60s)**: ~43 ms

This is comparable to the current total simulation time (9 ms with frozen kernel), so recomputation roughly 5x the cost. Still far faster than Fortran (82 ms).

## Decision

Defer implementation to Phase 5. The current JIT-compiled functions are designed to work inside `jax.lax.scan` with no modification — only the orchestration loop needs restructuring.

## Consequences

- Phase 1-2 validation uses frozen kernel (documented as caveat)
- Phase 3 (growth/nucleation) will change particle radii — at that point, either implement per-step recomputation or accept the approximation for validation
- Phase 5 will implement the full carry-state scan architecture
