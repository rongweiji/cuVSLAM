#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_PATH="${1:-}"

if [[ -z "${INPUT_PATH}" ]]; then
  if [[ -n "${CUVSLAM_LIB_PATH:-}" ]]; then
    INPUT_PATH="${CUVSLAM_LIB_PATH}"
  elif [[ -n "${CUVSLAM_LIBRARY_PATH:-}" ]]; then
    INPUT_PATH="${CUVSLAM_LIBRARY_PATH}"
  else
    cat <<'EOF' >&2
Usage:
  scripts/check_cuvslam.sh <path-to-libcuvslam.so|sdk-root|lib-dir>

Fallback inputs:
  CUVSLAM_LIB_PATH
  CUVSLAM_LIBRARY_PATH
EOF
    exit 2
  fi
fi

resolve_lib_path() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    printf '%s\n' "${path}"
    return 0
  fi
  if [[ -d "${path}" ]]; then
    if [[ -f "${path}/lib/libcuvslam.so" ]]; then
      printf '%s\n' "${path}/lib/libcuvslam.so"
      return 0
    fi
    if [[ -f "${path}/libcuvslam.so" ]]; then
      printf '%s\n' "${path}/libcuvslam.so"
      return 0
    fi
  fi
  return 1
}

LIB_PATH="$(resolve_lib_path "${INPUT_PATH}" || true)"
if [[ -z "${LIB_PATH}" ]]; then
  echo "Could not resolve libcuvslam.so from: ${INPUT_PATH}" >&2
  exit 2
fi

if [[ ! -f "${LIB_PATH}" ]]; then
  echo "libcuvslam.so not found: ${LIB_PATH}" >&2
  exit 2
fi

LIB_PATH="$(readlink -f "${LIB_PATH}")"
LIB_DIR="$(dirname "${LIB_PATH}")"
SDK_ROOT_CANDIDATE="$(cd "${LIB_DIR}/.." && pwd)"
SDK_ROOT=""
if [[ -f "${SDK_ROOT_CANDIDATE}/include/cuvslam/cuvslam2.h" ]]; then
  SDK_ROOT="${SDK_ROOT_CANDIDATE}"
fi

echo "Resolved libcuvslam: ${LIB_PATH}"
if [[ -n "${SDK_ROOT}" ]]; then
  echo "Resolved SDK root: ${SDK_ROOT}"
else
  echo "SDK headers not found beside library (optional for this project)." >&2
fi

if command -v nm >/dev/null 2>&1; then
  # Avoid false failures from pipefail when rg exits early after first match.
  set +o pipefail
  nm -D --defined-only "${LIB_PATH}" | rg -q "CUVSLAM_GetVersion"
  sym_ok=$?
  set -o pipefail
  if [[ ${sym_ok} -ne 0 ]]; then
    echo "Symbol check failed: CUVSLAM_GetVersion not found in ${LIB_PATH}" >&2
    exit 3
  fi
  echo "Symbol check passed: CUVSLAM_GetVersion"
elif command -v objdump >/dev/null 2>&1; then
  set +o pipefail
  objdump -T "${LIB_PATH}" | rg -q "CUVSLAM_GetVersion"
  sym_ok=$?
  set -o pipefail
  if [[ ${sym_ok} -ne 0 ]]; then
    echo "Symbol check failed: CUVSLAM_GetVersion not found in ${LIB_PATH}" >&2
    exit 3
  fi
  echo "Symbol check passed: CUVSLAM_GetVersion"
else
  echo "Skipping symbol check (nm/objdump not available)." >&2
fi

if command -v ldd >/dev/null 2>&1; then
  echo "Dependency check (ldd):"
  ldd "${LIB_PATH}" || true
  if ldd "${LIB_PATH}" | rg -q "not found"; then
    echo "Some shared-library dependencies are missing (see ldd output above)." >&2
    exit 4
  fi
fi

echo ""
echo "Recommended configure command:"
echo "cmake -S ${ROOT_DIR} -B ${ROOT_DIR}/build \\"
echo "  -DCMAKE_BUILD_TYPE=Release \\"
echo "  -DCUVSLAM_LIBRARY=${LIB_PATH} \\"
echo "  -DCUVSLAM_REQUIRE_LIBRARY=ON"
if [[ -n "${SDK_ROOT}" ]]; then
  echo ""
  echo "Alternative:"
  echo "cmake -S ${ROOT_DIR} -B ${ROOT_DIR}/build \\"
  echo "  -DCMAKE_BUILD_TYPE=Release \\"
  echo "  -DCUVSLAM_SDK_ROOT=${SDK_ROOT} \\"
  echo "  -DCUVSLAM_REQUIRE_LIBRARY=ON"
fi
