#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_rerun"
OUTPUT_DIR="${1:-${ROOT_DIR}/outputs/data_sample_rerun}"
LIB_INPUT="${2:-}"
REALTIME_SPEED="${3:-1.0}"
MAX_FRAMES="${4:-0}"
SPAWN_FLAG="${5:---spawn}"
LOG_EVERY_N="${6:-3}"

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

mkdir -p "${ROOT_DIR}/.tools/rerun" "${OUTPUT_DIR}"

if [[ ! -x "${ROOT_DIR}/.tools/rerun/rerun" ]]; then
  echo "Rerun CLI not found; installing local copy..."
  "${ROOT_DIR}/scripts/install_rerun_cli.sh" "${ROOT_DIR}/.tools/rerun"
fi

export PATH="${ROOT_DIR}/.tools/rerun:${PATH}"
RERUN_BIN="${ROOT_DIR}/.tools/rerun/rerun"

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DCUVSLAM_ENABLE_RERUN=ON
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
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"

CMD=(
  "${BUILD_DIR}/cuvslam_cli"
  --dataset_root "${ROOT_DIR}/data_sample"
  --dataset_format custom
  --output_dir "${OUTPUT_DIR}"
  --enable_rerun
  --rerun_save "${OUTPUT_DIR}/cuvslam_data_sample.rrd"
  --rerun_log_every_n "${LOG_EVERY_N}"
  --realtime
  --realtime_speed "${REALTIME_SPEED}"
  "${CLI_ARGS[@]}"
)

if [[ "${MAX_FRAMES}" != "0" ]]; then
  CMD+=(--max_frames "${MAX_FRAMES}")
fi

if [[ "${SPAWN_FLAG}" == "--spawn" ]]; then
  CMD+=(--rerun_spawn)
fi

"${CMD[@]}"

echo ""
echo "Performance report: ${OUTPUT_DIR}/performance_report.md"
sed -n '1,180p' "${OUTPUT_DIR}/performance_report.md"
echo ""
echo "Rerun file saved: ${OUTPUT_DIR}/cuvslam_data_sample.rrd"
echo "Open with: ${RERUN_BIN} ${OUTPUT_DIR}/cuvslam_data_sample.rrd"
