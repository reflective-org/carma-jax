#!/usr/bin/env bash
# Apply the Phase 10.2 Fortran ensemble patch to ../original-carma.
#
# Copies carma_sulfatetest_ensemble.F90 into CARMA/tests/, adds the
# CMake entry if missing, and rebuilds. Idempotent: safe to re-run.
#
# Usage:
#   ./scripts/fortran_patch/apply_patch.sh [ORIGINAL_CARMA_PATH]
#
# Default ORIGINAL_CARMA_PATH is ../original-carma (relative to repo root).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIGINAL_CARMA="${1:-${REPO_ROOT}/../original-carma}"

F90_SRC="${REPO_ROOT}/scripts/fortran_patch/carma_sulfatetest_ensemble.F90"
TESTS_DIR="${ORIGINAL_CARMA}/CARMA/tests"
CMAKELISTS="${TESTS_DIR}/CMakeLists.txt"
BUILD_DIR="${ORIGINAL_CARMA}/CARMA/build"

if [[ ! -d "${TESTS_DIR}" ]]; then
  echo "ERROR: Cannot find CARMA tests dir at ${TESTS_DIR}" >&2
  echo "       Override with: ./apply_patch.sh /path/to/original-carma" >&2
  exit 1
fi

echo "==> Copying F90 into ${TESTS_DIR}"
cp "${F90_SRC}" "${TESTS_DIR}/carma_sulfatetest_ensemble.F90"

CMAKE_LINE='create_standard_test(NAME sulfate_ensemble SOURCES atmosphere_mod.F90 carma_sulfatetest_ensemble.F90)'
if grep -qF "${CMAKE_LINE}" "${CMAKELISTS}"; then
  echo "==> CMake entry already present"
else
  echo "==> Appending CMake entry to ${CMAKELISTS}"
  # Prefix with newline — some CMakeLists don't end with one, so
  # a raw append would glue the entry onto the previous line.
  printf '\n%s\n' "${CMAKE_LINE}" >> "${CMAKELISTS}"
fi

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "ERROR: Build dir ${BUILD_DIR} missing. Run cmake first." >&2
  exit 1
fi

echo "==> Rebuilding in ${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake .. >/dev/null
make test_sulfate_ensemble

BINARY="${BUILD_DIR}/test_sulfate_ensemble"
if [[ -x "${BINARY}" ]]; then
  echo ""
  echo "SUCCESS. Binary at: ${BINARY}"
  echo ""
  echo "Test with:"
  echo "  cd ${REPO_ROOT}"
  echo "  python scripts/fortran_orchestrator.py \\"
  echo "    --binary ${BINARY} \\"
  echo "    --scenarios data/sulfate_scenarios_1000.npz \\"
  echo "    --out data/sulfate_fortran_outputs.npz"
else
  echo "ERROR: Build finished but binary not found at ${BINARY}" >&2
  exit 1
fi
