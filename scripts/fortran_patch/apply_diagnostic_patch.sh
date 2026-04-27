#!/usr/bin/env bash
# Apply the Phase 0 diagnostic-binary patch to ../original-carma.
# Mirrors apply_patch.sh but builds test_sulfate_diagnostic instead.
#
# The diagnostic binary writes per-substep dumps of every microfast/
# microslow intermediate array to a directory passed as argv[2].
# Used by tests/diff/ to validate JAX kernels against Fortran's actual
# state arrays.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIGINAL_CARMA="${1:-${REPO_ROOT}/../original-carma}"

F90_SRC="${REPO_ROOT}/scripts/fortran_patch/carma_sulfatetest_diagnostic.F90"
TESTS_DIR="${ORIGINAL_CARMA}/CARMA/tests"
CMAKELISTS="${TESTS_DIR}/CMakeLists.txt"
BUILD_DIR="${ORIGINAL_CARMA}/CARMA/build"

if [[ ! -d "${TESTS_DIR}" ]]; then
  echo "ERROR: Cannot find CARMA tests dir at ${TESTS_DIR}" >&2
  echo "       Override with: ./apply_diagnostic_patch.sh /path/to/original-carma" >&2
  exit 1
fi

echo "==> Copying F90 into ${TESTS_DIR}"
cp "${F90_SRC}" "${TESTS_DIR}/carma_sulfatetest_diagnostic.F90"

CMAKE_LINE='create_standard_test(NAME sulfate_diagnostic SOURCES atmosphere_mod.F90 carma_sulfatetest_diagnostic.F90)'
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
make test_sulfate_diagnostic

BINARY="${BUILD_DIR}/test_sulfate_diagnostic"
if [[ -x "${BINARY}" ]]; then
  echo ""
  echo "SUCCESS. Binary at: ${BINARY}"
  echo ""
  echo "Test with:"
  echo "  mkdir -p ${REPO_ROOT}/data/diff/scen_024"
  echo "  echo '213.2 155.4 0.27 0.14 35.0 1.6' > /tmp/scen024.txt"
  echo "  ${BINARY} /tmp/scen024.txt ${REPO_ROOT}/data/diff/scen_024"
  echo "  ls ${REPO_ROOT}/data/diff/scen_024/ | head"
else
  echo "ERROR: Build finished but binary not found at ${BINARY}" >&2
  exit 1
fi
