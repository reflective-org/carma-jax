#!/usr/bin/env bash
# Build the patched Fortran sulfate test for the realistic-ensemble benchmark.
#
# Copies benchmark_final/fortran/carma_sulfatetest_realistic.F90 into
# ../original-carma/CARMA/tests/, registers it with CMake (if not already
# done), and rebuilds. Idempotent: safe to re-run.
#
# Usage: ./benchmark_final/fortran/build_realistic.sh [ORIGINAL_CARMA_PATH]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIGINAL_CARMA="${1:-${REPO_ROOT}/../original-carma}"

F90_SRC="${REPO_ROOT}/benchmark_final/fortran/carma_sulfatetest_realistic.F90"
TESTS_DIR="${ORIGINAL_CARMA}/CARMA/tests"
CMAKELISTS="${TESTS_DIR}/CMakeLists.txt"
BUILD_DIR="${ORIGINAL_CARMA}/CARMA/build"

if [[ ! -d "${TESTS_DIR}" ]]; then
  echo "ERROR: Cannot find CARMA tests dir at ${TESTS_DIR}" >&2
  exit 1
fi

echo "==> Copying patched F90 into ${TESTS_DIR}"
cp "${F90_SRC}" "${TESTS_DIR}/carma_sulfatetest_realistic.F90"

CMAKE_LINE='create_standard_test(NAME sulfate_realistic SOURCES atmosphere_mod.F90 carma_sulfatetest_realistic.F90)'
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
make test_sulfate_realistic

BINARY="${BUILD_DIR}/test_sulfate_realistic"
if [[ -x "${BINARY}" ]]; then
  echo ""
  echo "SUCCESS. Binary at: ${BINARY}"
else
  echo "ERROR: build finished but binary missing at ${BINARY}" >&2
  exit 1
fi
