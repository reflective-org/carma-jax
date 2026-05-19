#!/usr/bin/env bash
# Apply the Phase 11.4 diagnostic patch — carma_nuc2test_diagnostic.
# Parallel to apply_diagnostic_patch.sh; same build flow.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIGINAL_CARMA="${1:-${REPO_ROOT}/../original-carma}"

F90_SRC="${REPO_ROOT}/scripts/fortran_patch/carma_nuc2test_diagnostic.F90"
TESTS_DIR="${ORIGINAL_CARMA}/CARMA/tests"
CMAKELISTS="${TESTS_DIR}/CMakeLists.txt"
BUILD_DIR="${ORIGINAL_CARMA}/CARMA/build"

if [[ ! -d "${TESTS_DIR}" ]]; then
  echo "ERROR: Cannot find CARMA tests dir at ${TESTS_DIR}" >&2
  exit 1
fi

echo "==> Copying F90 into ${TESTS_DIR}"
cp "${F90_SRC}" "${TESTS_DIR}/carma_nuc2test_diagnostic.F90"

CMAKE_LINE='create_standard_test(NAME nuc2test_diagnostic SOURCES atmosphere_mod.F90 carma_nuc2test_diagnostic.F90)'
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
make test_nuc2test_diagnostic

BINARY="${BUILD_DIR}/test_nuc2test_diagnostic"
if [[ -x "${BINARY}" ]]; then
  echo ""
  echo "SUCCESS. Binary at: ${BINARY}"
  echo ""
  echo "Test with:"
  echo "  mkdir -p /tmp/nuc2_test"
  echo "  echo '210.0 200.0 0.95 100.0 2.5e-6 1.5' > /tmp/scen_nuc2.txt"
  echo "  ${BINARY} /tmp/scen_nuc2.txt /tmp/nuc2_test 5"
  echo "  ls /tmp/nuc2_test/ | head"
else
  echo "ERROR: Binary not found at ${BINARY}" >&2
  exit 1
fi
