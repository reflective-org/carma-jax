#!/usr/bin/env bash
# Compile the two tiny Phase 11.6 standalone drivers.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${REPO_ROOT}/../original-carma/CARMA/build_standalone"
mkdir -p "${OUT_DIR}"

for k in freezdropl melticel; do
  SRC="${REPO_ROOT}/scripts/fortran_patch/${k}_standalone.F90"
  OUT_BIN="${OUT_DIR}/${k}_standalone"
  echo "==> gfortran -O2 -Wall ${SRC} -> ${OUT_BIN}"
  gfortran -O2 -Wall "${SRC}" -o "${OUT_BIN}"
done
echo "SUCCESS. Binaries in ${OUT_DIR}"
