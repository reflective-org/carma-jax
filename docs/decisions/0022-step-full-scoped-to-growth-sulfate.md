# ADR 0022: `make_step_full` composes growth + sulfate only; transport/coag stay external

## Status

Accepted — Phase 9.4.

## Context

The original Phase 9.4 plan says:

> Composes `make_step_transport` + `make_microslow` +
> `make_microfast_growth` + Phase 7 `make_step_sulfate` + Phase 8
> `fixcorecol`, all pre-bound to a `CarmaConfig`. Dispatches by
> `do_*` flags.

In practice, the four existing factories have divergent signatures:

| Factory | # call-time args |
|---|---|
| `make_step_transport` | 19 |
| `make_step_coag` | 9 |
| `make_step_microfast` | ~25 |
| `make_step_sulfate` | 6 |

A single flat factory would need to unify all 59+ arguments. That's
a design exercise worth doing once — but not before we know what
Phase 10's ensemble actually needs from the composed signature.

## Decision

Scope Phase 9.4's `make_step_full` to the microphysics composition
that was **not** already a single JIT closure:

- Adaptive-retry growth (`newstate_calc_growth_jit`, Phase 9.3)
- Sulfate nucleation + gas exchange (`sulfate_step_one_level`,
  Phase 7.6b)

Transport (`make_step_transport`) and coagulation (`make_step_coag`)
stay as independent factories. The column driver schedules them at
the appropriate outer-loop position:

```python
step_transport = make_step_transport(config)
step_coag      = make_step_coag(config)
step_full      = make_step_full(config)           # growth + sulfate

for istep in range(nstep):
    pc, gc, t, ...       = step_transport(...)
    pc, gc, t, ...       = step_coag(...)
    pc, gc, t, diag      = step_full(pc, gc, t, ..., dtime)
```

This mirrors Fortran's `step.F90` → `vertical → microslow → newstate`
chain, with each stage its own JIT closure. Phase 10 can decide
whether the between-stage Python boundary is a measurable cost and
promote into a flat factory later.

## Alternatives considered

- **Unified flat factory now.** Rejected — 59-arg signature is a
  reviewing nightmare; the benefit (removing Python boundaries)
  isn't yet measured. Phase 10 benchmark will quantify if the
  boundary matters.
- **Closure-of-closures with a unified state NamedTuple.** The path
  Phase 10 likely takes if boundaries show up as hot. Deferred to
  avoid doing a design exercise that might be wrong.
- **Drop make_step_full entirely.** Rejected — growth + sulfate is
  a natural single closure (they share per-column state and get
  called in immediate sequence).

## Consequences

- ``step_full`` is a single ``@jax.jit`` closure: 0.09 ms per warm
  call single-column, 0.002 ms per column under vmap-1000 on CPU.
- 5 unit tests verify the factory runs, nucleates, disables cleanly,
  conserves mass, and reproduces the same output on warm calls.
- Benchmarks (``scripts/benchmark_step_full.py``) report:
  - Cold trace+compile: 1.4 s (amortised across all subsequent calls)
  - Warm single call: 92 μs
  - vmap batch=1000: 2.1 ms total / 2 μs per column
- The transport/coag/sulfate orchestration pattern is documented in
  Phase 10's ensemble harness.

## Validation

- 5 unit tests pass.
- Full suite: 190/190 unit tests pass.
- Benchmark figure committed to ``plots/phase9_step_full/``.
