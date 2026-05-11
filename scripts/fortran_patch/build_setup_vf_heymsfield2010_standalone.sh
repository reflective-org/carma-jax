#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${REPO_ROOT}/scripts/fortran_patch/setup_vf_heymsfield2010_standalone.F90"
OUT_DIR="${REPO_ROOT}/../original-carma/CARMA/build_standalone"
mkdir -p "${OUT_DIR}"
OUT_BIN="${OUT_DIR}/setup_vf_heymsfield2010_standalone"
echo "==> gfortran -O2 -Wall ${SRC} -> ${OUT_BIN}"
gfortran -O2 -Wall "${SRC}" -o "${OUT_BIN}"
echo "SUCCESS. Binary at: ${OUT_BIN}"
