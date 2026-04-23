"""Mock Fortran sulfatetest binary (Phase 10.2).

Reads a scenario from ``argv[1]`` and writes a JSON result to
``argv[2]`` — mimicking the interface the real Fortran binary will
honor once the modifications from ADR 0025 land.

This mock produces physically-plausible-but-not-calibrated outputs:
it runs a very simple Euler integration of dG/dt = -k · G on the
H2SO4 gas, so multi-scenario orchestrator tests can verify that
different scenarios produce different outputs without needing the
real Fortran build.
"""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("usage: _mock_fortran_sulfate.py <scenario> <output>",
              file=sys.stderr)
        sys.exit(2)

    scen_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    values = scen_path.read_text().split()
    T, p, rh, h2so4_pptv, mu_nm, sigma_g = [float(v) for v in values[:6]]

    # Simple "physics": gas decays at rate ∝ rh·h2so4 when cold; 100 steps
    nstep = 100
    dt = 60.0
    k = 0.01 * rh * (1.0 if T < 250 else 0.1)
    h2so4 = h2so4_pptv
    pc_final = [h2so4_pptv * 0.01 / (1 + i) for i in range(38)]   # 38 bins
    for _ in range(nstep):
        h2so4 *= max(0.0, 1.0 - k * dt / 3600.0)

    out = dict(
        T_final=T,
        gc_h2so4_final=h2so4,
        pc_final=pc_final,
        nstep_ran=nstep,
        status="ok",
    )
    out_path.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
