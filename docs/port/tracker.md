# CARMA-JAX port tracker

Function-by-function status for the CARMA Fortran→JAX port.
Each row is a single function with a Fortran-bench gate.

The tracker is the operational artifact we work against. The high-level roadmap lives in `docs/ROADMAP.md` (and the plan file at `~/.claude/plans/sprightly-waddling-stardust.md`).

## How to read this file

| Box | Meaning |
|-----|---------|
| `[ ]` | not yet bench-tested against Fortran |
| `[/]` | bench fails — discrepancy under discussion |
| `[x]` | bench passes — JAX matches Fortran (rtol ≤ 1e-10 unless noted) |
| `[-]` | not applicable for the current sulfate-test scope (e.g. ice-only paths) |

Each Fortran↔JAX file pairing is a section. Rows under it are functions inside the file.
A function is "done" only when the bench gate passes against `data/diff/scen_<NNN>/` dumps from the instrumented Fortran binary.

**Workflow per row:**
1. Read Fortran source for the function.
2. Read JAX implementation.
3. Run the diff harness for that function on scenario 24, substep 1.
4. **Match → tick `[x]`, commit, next.**
5. **Mismatch → flip to `[/]`, write up the diff, pause and discuss with user before any fix.**

---

## Phase 0 — Diagnostic infrastructure

- [x] `scripts/fortran_patch/carma_sulfatetest_diagnostic.F90` — instrumented binary that dumps per-substep state
- [x] `scripts/fortran_patch/apply_diagnostic_patch.sh` — patch script (mirrors apply_patch.sh)
- [x] `src/carma/_diag/fortran_reader.py` — Python loader for Fortran unformatted-stream dumps
- [x] `tests/diff/test_kernel_diff.py` — diff harness with `assert_kernel_match` helper + `make_kernel_plot` helper
- [x] **Phase 0 exit gate:** `vaporp_h2o_murphy2005` bench passes against scen 24 dump (max rel err = 0.0)
- [x] **Phase 0.5:** `scripts/run_diagnostic_ensemble.py` — orchestrator dumps all 1000 scenarios at substep 1 in ~5 s
- [x] **Phase 0.5:** parameterized tests across all dumped scenarios + per-kernel summary CDF/histogram in `plots/diff/summary/`

---

## Phase 7 — H₂SO₄ sulfate physics

### `src/carma/sulfate_utils.py`  ↔  `sulfate_utils.F90`
- [x] `wtpct_tabaz` (1000/1000 scenarios pass at rtol 1e-10; ~750 at machine ε, ~250 at ~1e-16)
- [x] `sulfate_density` (1000/1000 scenarios pass at rtol 1e-10)
- [x] `sulfate_surf_tens` (1000/1000 scenarios pass at rtol 1e-10)

### `src/carma/vapor_pressure.py`  ↔  `vaporp_*.F90`
- [x] `vaporp_h2o_murphy2005` (1000/1000 scenarios pass at rtol 1e-10; ~750 at machine ε, ~250 at ~1e-16)
- [x] `vaporp_h2so4_ayers1980` (1000/1000 scenarios pass at rtol 1e-10; ~720 at machine ε, max 2.1e-14). Fix: call `wtpct_tabaz` instead of the crude `100*(1-rh)*0.98` approximation.
- [-] `vaporp_h2o_buck1981` (sulfate test uses Murphy 2005 only; not exercised by current bench)
- [-] `vaporp_h2o_goff1946` (sulfate test uses Murphy 2005 only; not exercised by current bench)

### `src/carma/supersaturation.py`  ↔  `supersat.F90`
- [x] `supersat` (4 × 1000/1000 pass at rtol 1e-10 across {supsatl, supsati} × {h2o, h2so4}; ~960 at machine ε for most paths, supsati[h2o] max 5.8e-13). Sulfate test is clearsky; `supersat_with_cloud` not exercised.

### `src/carma/wetr.py`  ↔  `wetr.F90`
- [ ] `_wetr_petters`
- [ ] `_wetr_wtpct`
- [ ] `get_wetr` (dispatch)

### `src/carma/hygroscopicity.py`  ↔  `hygroscopicity.F90`
- [ ] `hygroscopicity`

### `src/carma/rhopart.py`  ↔  `rhopart.F90`
- [ ] `rhopart`

### `src/carma/nucleation/sulfnucrate.py`  ↔  `sulfnucrate.F90`
- [ ] `binary_nuc_zhao1995`
- [ ] `binary_nuc_vehk2002`

### `src/carma/nucleation/sulfhetnucrate.py`  ↔  `sulfhetnucrate.F90`
- [ ] `sulfhetnucrate`

### `src/carma/nucleation/sulfnuc.py`  ↔  `sulfnuc.F90`
- [ ] `homogeneous_nucleation`
- [ ] `heterogeneous_nucleation`
- [ ] `sulfnuc` (driver)

### `src/carma/gasexchange.py`  ↔  `gasexchange.F90`
- [ ] `gasexchange`

---

## Phase 8 — Correctness primitives

### `src/carma/utils/smallconc.py`  ↔  `smallconc.F90` + `maxconc.F90`
- [ ] `smallconc`
- [ ] `maxconc`

### `src/carma/utils/fixcorecol.py`  ↔  `fixcorecol.F90`
- [ ] `fixcorecol`

### `src/carma/utils/coremasscheck.py`  ↔  `coremasscheck.F90`
- [ ] `coremasscheck`

### `src/carma/growth/pheat.py`  ↔  `pheat.F90`
- [ ] `pheat`

### `src/carma/growth/growevapl.py`  ↔  `growevapl.F90`
- [ ] `growevapl`

### `src/carma/growth/growp.py`  ↔  `growp.F90`
- [ ] `growp`

### `src/carma/growth/upgxfer.py`  ↔  `upgxfer.F90`
- [ ] `upgxfer`

### `src/carma/growth/evapp.py`  ↔  `evapp.F90` + `evap_ingrp.F90`
- [ ] `evapp`
- [ ] `downgevapply`

### `src/carma/growth/downgxfer.py`  ↔  `downgxfer.F90`
- [ ] `downgxfer`

### `src/carma/growth/evap_mono.py` + `evap_poly.py`  ↔  `evap_mono.F90` + `evap_poly.F90`
- [ ] `evap_mono`
- [ ] `evap_poly`

### `src/carma/solvers/psolve.py`  ↔  `psolve.F90`
- [ ] `psolve`

### `src/carma/solvers/gsolve.py`  ↔  `gsolve.F90`
- [ ] `gsolve`

### `src/carma/solvers/tsolve.py`  ↔  `tsolve.F90`
- [ ] `tsolve`

### `src/carma/solvers/totalcondensate.py`  ↔  `totalcondensate.F90`
- [ ] `totalcondensate`

### `src/carma/nsubsteps.py`  ↔  `nsubsteps.F90`
- [ ] `nsubsteps`

### `src/carma/prestep.py`  ↔  `prestep.F90`
- [ ] `prestep`

### Coag pipeline (instrumented in Phase 0 per user direction)

### `src/carma/coagulation/setup_coag.py`  ↔  `setupcoag.F90` + `setupckern.F90`
- [ ] `setup_coag` (kbin, volx, pkernel)

### `src/carma/setup_ckern.py`  ↔  `setupckern.F90`
- [ ] `setup_ckern` (Brownian + grav coag kernel)

### `src/carma/coagulation/coagl.py`  ↔  `coagl.F90`
- [ ] `coagl` (loss rate)

### `src/carma/coagulation/coagp.py`  ↔  `coagp.F90`
- [ ] `coagp` (production rate)

### `src/carma/coagulation/csolve.py`  ↔  `csolve.F90`
- [ ] `csolve` (coag solver)

### `src/carma/microslow.py`  ↔  `microslow.F90`
- [ ] `microslow` (composition)

---

## Phase 9 — `step_full` / microfast composition

### `src/carma/newstate_calc.py`  ↔  `newstate_calc.F90`
- [ ] `microfast_growth` (composed; bench against full microfast.F90 output per substep)

### `src/carma/newstate_calc_jit.py`  ↔  `newstate_calc.F90` (retry block)
- [ ] adaptive retry loop (`nretries`, `nsubsteps` decisions match Fortran)

### `src/carma/newstate.py`  ↔  `newstate.F90`
- [ ] `newstate` dispatcher (clearsky / incloud)

### `src/carma/step_full.py`  ↔  `step.F90`
- [ ] `make_step_full` composition

### `src/carma/detrain.py`  ↔  `detrain.F90`
- [-] `detrain` (cloud-only — N/A for sulfate test)

---

## Phase 10 — 1000-scenario ensemble

- [ ] re-run Fortran ensemble (no changes; just re-confirm)
- [ ] re-run JAX ensemble through bench-validated `step_full`
- [ ] per-bin distribution plots (`fig2_per_bin_distributions.png` etc.)
- [ ] statistical comparison: ≥ 95% scenarios match ≤ 1% median; max ≤ 5%

---

## Phase 11 — Cloud / ice

### `src/carma/nucleation/actdropl.py`  ↔  `actdropl.F90`
- [ ] `actdropl`

### `src/carma/nucleation/freezdropl.py`  ↔  `freezdropl.F90`
- [ ] `freezdropl`

### `src/carma/nucleation/melticel.py`  ↔  `melticel.F90`
- [ ] `melticel`

### `src/carma/nucleation/hetnucl.py`  ↔  `hetnucl.F90`
- [ ] `hetnucl`

### `src/carma/nucleation/freezaerl_mohler2010.py`  ↔  `freezaerl_mohler2010.F90`
- [ ] `freezaerl_mohler2010`

### `src/carma/nucleation/freezaerl_tabazadeh2000.py`  ↔  `freezaerl_tabazadeh2000.F90`
- [ ] `freezaerl_tabazadeh2000`

### `src/carma/rhoice_heymsfield2010.py`  ↔  `rhoice_heymsfield2010.F90`
- [ ] `rhoice_heymsfield2010`

### `src/carma/setup_vf.py`  ↔  `setupvf*.F90` family
- [ ] `setupvf_std`
- [ ] `setupvf_std_shape`
- [ ] `setupvf_heymsfield2010`

---

## Bench tolerance

Default: `rtol=1e-10` (very strict; catches any reordering or unit issue).
Per-kernel override: maintain a small map in `tests/diff/test_kernel_diff.py` for kernels with legitimate floating-point reordering vs Fortran (e.g. summation-order differences in vectorized JAX).

## Bench scenarios

Phase 0 dumps span **all 1000 scenarios** at substep 1 (~5 s wall, ~109 MB). Each kernel test parameterizes over all of them and asserts every scenario meets the rtol. Per-kernel summary plots in `plots/diff/summary/` show the histogram + CDF of max relative error across scenarios.

If we need deeper substep coverage on a specific kernel, regenerate dumps for that scenario with `--nstep-max N` (or all scenarios with N substeps).

## Dump location

`data/diff/scen_<NNN>/substep_<NNNN>_<arrayname>.bin` — one folder per scenario, one file per (substep, array). Fortran unformatted stream format.
