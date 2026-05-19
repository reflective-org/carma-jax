"""Generate a stratified ensemble of sulfate-microphysics scenarios.

Phase 10.1 — produces ``(T, p, RH, H2SO4, aerosol_mu, aerosol_sigma_g)``
tuples spanning the stratosphere-to-troposphere envelope. Uses
``scipy.stats.qmc.LatinHypercube`` for low-discrepancy sampling so
1000 scenarios cover the parameter hypercube uniformly without
random-clumping artefacts.

Sampling envelope (matches the plan):

| param | range | units |
|---|---|---|
| T | [180, 310] | K |
| p | [1, 1013] | hPa |
| RH | [0.01, 1.05] | (fraction) |
| H2SO4 | [0.01, 100] | pptv (log-uniform) |
| aerosol mode μ | [5, 500] | nm (log-uniform) |
| aerosol σ_g | [1.2, 2.5] | (geometric SD) |

The H2SO4 and aerosol-mu axes are log-uniform because the dynamic
range spans 4 and 2 orders of magnitude respectively; sampling
log-uniformly gives equal coverage of the orders rather than
clumping the high end.

Default output: ``data/sulfate_scenarios_1000.npz`` with arrays
``T``, ``p``, ``rh``, ``h2so4_pptv``, ``aerosol_mu_nm``,
``aerosol_sigma_g``, plus metadata.
"""

import argparse
from pathlib import Path

import numpy as np


# Sampling bounds (kept module-level so unit tests can import them)
_BOUNDS = {
    "T":              (180.0, 310.0),       # K, linear
    "p":              (1.0, 1013.0),         # hPa, linear
    "rh":             (0.01, 1.05),          # fraction, linear
    "h2so4_pptv":     (0.01, 100.0),         # pptv, LOG
    "aerosol_mu_nm":  (5.0, 500.0),          # nm, LOG
    "aerosol_sigma_g": (1.2, 2.5),           # geometric SD, linear
}

# Phase 10.4c: realistic stratospheric envelope. Narrowed to UTLS
# conditions where stratospheric sulfate physics is actually
# meaningful, with low background aerosol so particles don't grow
# into the top bin (where Fortran/JAX top-bin handling differs).
_BOUNDS_REALISTIC = {
    "T":              (200.0, 240.0),       # K, UTLS / lower stratosphere
    "p":              (50.0, 200.0),         # hPa, UTLS
    "rh":             (0.10, 0.80),          # avoid extremes
    "h2so4_pptv":     (0.01, 5.0),           # background to mildly perturbed
    "aerosol_mu_nm":  (10.0, 100.0),         # Aitken / fresh nuc mode (NOT coarse)
    "aerosol_sigma_g": (1.4, 1.8),           # typical stratospheric width
}

_LOG_AXES = ("h2so4_pptv", "aerosol_mu_nm")


def generate_scenarios(n: int, seed: int = 42, realistic: bool = False):
    """Return a dict of named arrays of length ``n``.

    Args:
        n: number of scenarios.
        seed: PRNG seed for reproducibility.
        realistic: if True, use the narrow stratospheric envelope
            (Phase 10.4c). Default False = full hypercube (Phase 10.1).

    Returns:
        Dict with one ``(n,)`` numpy array per parameter, plus a
        ``"_seed"`` int and ``"_n"`` int for metadata.
    """
    from scipy.stats import qmc

    bounds = _BOUNDS_REALISTIC if realistic else _BOUNDS

    sampler = qmc.LatinHypercube(d=len(bounds), seed=seed)
    u = sampler.random(n)                     # (n, d) in [0, 1]

    out = {}
    for k, (lo, hi) in bounds.items():
        col = u[:, list(bounds).index(k)]
        if k in _LOG_AXES:
            out[k] = np.exp(np.log(lo) + col * (np.log(hi) - np.log(lo)))
        else:
            out[k] = lo + col * (hi - lo)
    out["_seed"] = seed
    out["_n"] = n
    return out


def save_scenarios(scenarios: dict, path: Path):
    """Save scenario dict as compressed NPZ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **scenarios)


def load_scenarios(path: Path) -> dict:
    """Load NPZ produced by ``save_scenarios``."""
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def main():
    parser = argparse.ArgumentParser(description="Generate sulfate scenarios")
    parser.add_argument("-n", type=int, default=1000,
                        help="Number of scenarios (default 1000)")
    parser.add_argument("-o", "--out", type=Path,
                        default=Path("data/sulfate_scenarios_1000.npz"),
                        help="Output NPZ path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--realistic", action="store_true",
                        help="Use narrow stratospheric envelope")
    args = parser.parse_args()

    label = "realistic" if args.realistic else "full hypercube"
    print(f"Generating {args.n} scenarios with seed {args.seed} ({label})...")
    scenarios = generate_scenarios(args.n, seed=args.seed,
                                     realistic=args.realistic)
    save_scenarios(scenarios, args.out)

    print(f"Saved to {args.out}")
    bounds = _BOUNDS_REALISTIC if args.realistic else _BOUNDS
    for k in bounds:
        arr = scenarios[k]
        print(f"  {k:18s}: min={arr.min():.4g}  max={arr.max():.4g}"
              f"  mean={arr.mean():.4g}")


if __name__ == "__main__":
    main()
