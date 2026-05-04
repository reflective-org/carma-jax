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
- [-] `_wetr_petters` (sulfate test uses `I_WTPCT_H2SO4`; Petters branch not exercised — defer to ice/cloud phase)
- [x] `_wetr_wtpct` (1000/1000 pass at rtol 1e-8; median 3.6e-11, p99 1.2e-9, max 2.3e-9). Cube roots use `jnp.cbrt`. Per-kernel tol relaxed to 1e-8 because the Kelvin-iteration chain (`cbrt → exp → wtpct_tabaz → sulfate_density → cbrt`) accumulates sub-ULP reordering past 1e-10; cbrt vs `**(1/3)` made no measurable difference. Phase 0 diagnostic extended to dump `relhum`, `rhop`, `r_wet`, `rhop_wet`, `r_bin`, `rmass_bin`.
- [x] `get_wetr` (dispatch — for `I_WTPCT_H2SO4` it forwards directly to `_wetr_wtpct`; bench-validated above)

### `src/carma/hygroscopicity.py`  ↔  `hygroscopicity.F90`
- [-] `hygroscopicity` — gated on `irhswell == I_PETTERS` (hygroscopicity.F90:42); sulfate test uses I_WTPCT_H2SO4, so the kernel never runs in this bench scope. Defer to ice/cloud phase together with `_wetr_petters`.

### `src/carma/rhopart.py`  ↔  `rhopart.F90`
- [x] `rhopart` (1000/1000 pass at rtol 1e-10, bit-exact). Sulfate test exercises only the no-core path (`ncore=0` → `rhop = rhoelem = 1.923`). Multi-element / core-mass-truncation branches will be re-benched when other elements appear in later phases.

### `src/carma/nucleation/sulfnucrate.py`  ↔  `sulfnucrate.F90`
- [x] `binary_nuc_zhao1995` (1000/1000 pass at rtol 1e-10 across {nucrate_cgs, mass_cluster_dry, rstar, ftry}; max ~2e-13, median ~1e-14, near machine ε). Bench compares against a new `dump_zhao1995_probe` in the diagnostic that calls Fortran's `binary_nuc_zhao1995` at end-of-step state — bypasses the multi-substep gsolve evolution that decouples dumped `rhompe` from dumped `gc`/`t`. Diagnostic also dumps `rmassup_bin` and `rmrat_group`.
- [-] `binary_nuc_vehk2002` (sulfate test runs `sulfnucl_method='ZhaoTurco'`; Vehkamaki branch never invoked — defer to a test that selects it)

### `src/carma/nucleation/sulfhetnucrate.py`  ↔  `sulfhetnucrate.F90`
- [-] `sulfhetnucrate` — gated in `sulfnuc.F90:106` on `inucproc(iepart, ienucto) == I_HETNUCSULF`. Sulfate test calls `CARMA_AddNucleation(carma, 1, 1, I_HOMNUC, ...)` only (carma_sulfatetest.F90:163), never adds an `I_HETNUCSULF` mapping → kernel never invoked. Verified empirically: `rnuclg` is identically zero across all 1000 dumped scenarios. Defer to a future test that adds `I_HETNUCSULF`.

### `src/carma/nucleation/sulfnuc.py`  ↔  `sulfnuc.F90`
- [x] `homogeneous_nucleation` (1000/1000 pass at rtol 1e-10; max 2e-13, median 1e-14, near ε; nucbin matches Fortran 1000/1000). Bench reconstructs `expected_rhompe` from the Phase 7.9 zhao1995 probe + dumped `rmassup_bin`/`rmrat_group`.
- [-] `heterogeneous_nucleation` (calls `sulfhetnucrate`, gated out for sulfate test as per Phase 7.10 row — never invoked).
- [x] `sulfnuc` (driver) — 1000/1000 pass; rhompe matches near ε, rnuclg is identically 0 (matches Fortran since heterogeneous path is disabled in the sulfate scope). Tested with `do_heterogeneous=False` to mirror the Fortran I_HETNUCSULF gate.

### `src/carma/gasexchange.py`  ↔  `gasexchange.F90`
- [x] `gasexchange` (1000/1000 pass at rtol 1e-10; max 2.2e-11, median bit-exact). **Do not use in integration path.** Fortran's `gasexchange` is dead code (`microfast.F90:152` commented out — gsolve overwrites `gasprod` via total-condensate, see `gsolve.F90:52-58`). Bench validates the kernel via a new `dump_gasexchange_probe` that calls Fortran's gasexchange directly with end-of-step cstate. `sulfate_step.py` now derives `gasprod_h2so4 = -Δgc/dtime` (gsolve convention) instead of calling `gasexchange()`. Diagnostic also dumps `cmf` and `totevap` (both identically zero/false in the sulfate scope).

---

## Phase 8 — Correctness primitives

### `src/carma/utils/smallconc.py`  ↔  `smallconc.F90` + `maxconc.F90`
- [x] `smallconc` (1000/1000 bit-exact). Sulfate test has NELEM=1, itype=I_VOLATILE (number element only — no core-mass/second-moment elements), so only the `max(pc, SMALL_PC)` branch runs. All dumped pc values exceed SMALL_PC, so smallconc is a no-op; bench verifies idempotency.
- [x] `maxconc` (1000/1000 bit-exact). Probe needed: the regular `f_pconmax` dump is pre-microfast (newstate_calc.F90:175) while the dumped `pc` is post-microfast — not a matched pair. `dump_maxconc_probe` calls Fortran's maxconc at end-of-step state and dumps the result.

### `src/carma/utils/fixcorecol.py`  ↔  `fixcorecol.F90`
- [-] `fixcorecol` — gated on `ncore(igroup) > 0` (fixcorecol.F90:55); sulfate test has a single volatile element with no cores → entire routine is skipped. Verified: no negative `pc` values across all 1000 scenarios. Defer to a future multi-element test.

### `src/carma/utils/coremasscheck.py`  ↔  `coremasscheck.F90`
- [-] `coremasscheck` — gated on `ncore(igroup) > 0` (coremasscheck.F90:48); same reason as fixcorecol. Defer.

### `src/carma/growth/pheat.py`  ↔  `pheat.F90`
- [x] `pheat` (37×1000 = 37 000 (scen,bin) pairs pass at rtol 1e-10; median bit-exact, max 9.9e-14). Sulfate test: `do_pheat=False`, `NWAVE=0` → no-radiation branch only (`akas = exp(akelvin/rup_wet)` + standard Köhler dmdt). Probe calls Fortran's `pheat` for `ibin=1..NBIN-1` (matching growevapl's loop). Diagnostic now also dumps `rup_wet` (NZ,NBIN,NGROUP) used by the Kelvin factor.

### `src/carma/growth/growevapl.py`  ↔  `growevapl.F90`
- [x] `growevapl` (1000/1000 pass at rtol 1e-10; growlg max 1.8e-13 / evaplg max 9.9e-14, both medians near ε). **Bug fixed**: JAX was missing the `pc > SMALL_PC` gate on the growth/evaporation flux branches (`growevapl.F90:212-226`), causing non-zero evaplg on bins where `pc ≤ 1e-50` (Fortran zeros those out). Probe: `dump_growevapl_probe` calls maxconc then growevapl at end-of-step state. Diagnostic also dumps PPM tables (dm, pratt, prat, pden1, palr, igrowgas) and rup_wet.

### `src/carma/growth/growp.py`  ↔  `growp.F90`
- [x] `growp` (1000/1000 pass at rtol 1e-10; max 1.8e-13, median ~7e-16, near machine ε). Probe: `dump_growp_probe` runs `maxconc → growevapl → growp` chain at end-of-step state and dumps `growpe`. Trivial kernel — `growpe[ibin,ielem] = pc[ibin-1,ielem] * growlg[ibin-1,igroup]` with `pconmax > FEW_PC` gate.

### `src/carma/growth/upgxfer.py`  ↔  `upgxfer.F90`
- [-] `upgxfer` — gated on `rnuclg(ifrom,igfrom,igroup) > 0` (upgxfer.F90:104). Sulfate test never invokes heterogeneous nucleation (Phase 7.10) so `rnuclg = 0` everywhere; upgxfer's `rnucpe` accumulator is never updated. Verified empirically: `rnucpe = 0` across all 1000 scenarios.

### `src/carma/growth/evapp.py`  ↔  `evapp.F90` + `evap_ingrp.F90`
- [x] `evapp` (1000/1000 pass at rtol 1e-10; max 9.9e-14, median bit-exact). Sulfate test: NELEM=1, no cores → falls into `ic1==0` branch → calls `evap_ingrp` per bin (`evappe[ibin-1] += pc[ibin] * evaplg[ibin]`). evap_mono and evap_poly are core-only paths (see Phase 8.9).
- [x] `downgevapply` (in `src/carma/growth/evapp.py`; matches `downgevapply.F90`) — 1000/1000 pass at rtol 1e-10, max 9.4e-15, median bit-exact. Explicit Euler `pc += dt * (evappe + rnucpe)` then `smallconc` floor. Probe dumps both `pc_predowng_probe` and `pc_postdowng_probe`.

### `src/carma/growth/downgxfer.py`  ↔  `downgxfer.F90`
- [-] `downgxfer` — gated on `rnuclg(ifrom,igfrom,igroup) > 0` (downgxfer.F90:106), same as upgxfer (Phase 8.7). Sulfate test never invokes heterogeneous nucleation, so `rnuclg = 0` and `rnucpe` accumulator never updates. Defer to a test scenario that adds I_HETNUCSULF.

### `src/carma/growth/evap_mono.py` + `evap_poly.py`  ↔  `evap_mono.F90` + `evap_poly.F90`
- [-] `evap_mono` — invoked from `evapp.F90:153,179` only when `ic1 != 0` (group has cores) AND `evap_total` is set. Sulfate test has `ncore=0` → `ic1=0` → these branches are unreachable. Defer to multi-element test.
- [-] `evap_poly` — same gate (called from `evapp.F90:181`). Defer.

### `src/carma/solvers/psolve.py`  ↔  `psolve.F90`
- [x] `psolve` (1000/1000 pass at rtol 1e-10; max 4.0e-15, median ~6.5e-16, near machine ε). **Sequential dependency**: Fortran's microfast loop `do ielem; do ibin; growp; upgxfer; psolve` has growp at bin `i` reading `pc[i-1]` AFTER psolve at `i-1` already updated it. The bench mirrors that loop order — pre-computing growpe in advance gave bin-3+ drift. Probe `dump_psolve_probe` runs the full prefix `maxconc → sulfnuc → growevapl → per-(ie,ib){growp,upgxfer,psolve}` at end-of-step state and dumps `pc_prepsolve_probe`, `rhompe_probe`, `pc_postpsolve_probe`.

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
