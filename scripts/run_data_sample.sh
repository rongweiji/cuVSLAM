#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
OUTPUT_DIR="${1:-${ROOT_DIR}/outputs/data_sample}"
LIB_INPUT="${2:-}"

# On WSL, prefer NVIDIA's driver-provided CUDA loader path.
if [[ -d "/usr/lib/wsl/lib" ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
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

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DCUVSLAM_REQUIRE_LIBRARY=ON
)

CLI_ARGS=()
CHECK_INPUT=""

if [[ -n "${LIB_INPUT}" ]]; then
  LIB_PATH="$(resolve_lib_path "${LIB_INPUT}" || true)"
  if [[ -z "${LIB_PATH}" ]]; then
    echo "Could not resolve libcuvslam from: ${LIB_INPUT}" >&2
    echo "Expected one of: /path/to/libcuvslam.so, /path/to/sdk_root, /path/to/lib_dir" >&2
    exit 2
  fi
  LIB_PATH="$(readlink -f "${LIB_PATH}")"
  CMAKE_ARGS+=("-DCUVSLAM_LIBRARY=${LIB_PATH}")
  CLI_ARGS+=(--libcuvslam_path "${LIB_PATH}")
  CHECK_INPUT="${LIB_PATH}"
elif [[ -n "${CUVSLAM_LIB_PATH:-}" ]]; then
  CHECK_INPUT="${CUVSLAM_LIB_PATH}"
elif [[ -n "${CUVSLAM_LIBRARY_PATH:-}" ]]; then
  CHECK_INPUT="${CUVSLAM_LIBRARY_PATH}"
fi

if [[ -n "${CHECK_INPUT}" ]]; then
  "${ROOT_DIR}/scripts/check_cuvslam.sh" "${CHECK_INPUT}"
else
  echo "No explicit libcuvslam path provided."
  echo "Relying on CMake autodiscovery (ament index / default linker paths)."
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"

mkdir -p "${OUTPUT_DIR}"

"${BUILD_DIR}/cuvslam_cli" \
  --dataset_root "${ROOT_DIR}/data_sample" \
  --dataset_format custom \
  --output_dir "${OUTPUT_DIR}" \
  --no_cuda \
  "${CLI_ARGS[@]}"

echo ""
echo "Generated report: ${OUTPUT_DIR}/performance_report.md"
sed -n '1,140p' "${OUTPUT_DIR}/performance_report.md"
