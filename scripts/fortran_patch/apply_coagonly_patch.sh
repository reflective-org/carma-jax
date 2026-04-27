#!/usr/bin/env bash
# Apply the coag-only ensemble Fortran patch (Phase 10.4k diagnostic).
#
# Copies carma_coagonly_ensemble.F90 into CARMA/tests/, adds the
# CMake entry if missing, and rebuilds. Idempotent: safe to re-run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIGINAL_CARMA="${1:-${REPO_ROOT}/../original-carma}"

F90_SRC="${REPO_ROOT}/scripts/fortran_patch/carma_coagonly_ensemble.F90"
TESTS_DIR="${ORIGINAL_CARMA}/CARMA/tests"
CMAKELISTS="${TESTS_DIR}/CMakeLists.txt"
BUILD_DIR="${ORIGINAL_CARMA}/CARMA/build"

if [[ ! -d "${TESTS_DIR}" ]]; then
  echo "ERROR: Cannot find CARMA tests dir at ${TESTS_DIR}" >&2
  exit 1
fi

echo "==> Copying F90 into ${TESTS_DIR}"
cp "${F90_SRC}" "${TESTS_DIR}/carma_coagonly_ensemble.F90"

CMAKE_LINE='create_standard_test(NAME coagonly_ensemble SOURCES atmosphere_mod.F90 carma_coagonly_ensemble.F90)'
if grep -qF "${CMAKE_LINE}" "${CMAKELISTS}"; then
  echo "==> CMake entry already present"
else
  echo "==> Appending CMake entry to ${CMAKELISTS}"
  printf '\n%s\n' "${CMAKE_LINE}" >> "${CMAKELISTS}"
fi

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "ERROR: Build dir ${BUILD_DIR} missing. Run cmake first." >&2
  exit 1
fi

echo "==> Rebuilding in ${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake .. >/dev/null
make test_coagonly_ensemble

BINARY="${BUILD_DIR}/test_coagonly_ensemble"
if [[ -x "${BINARY}" ]]; then
  echo ""
  echo "SUCCESS. Binary at: ${BINARY}"
else
  echo "ERROR: Build finished but binary not found at ${BINARY}" >&2
  exit 1
fi
