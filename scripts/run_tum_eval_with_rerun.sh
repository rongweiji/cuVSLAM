#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_rerun"
DATASET_ROOT="${1:-${ROOT_DIR}/third_party_datasets/tum_rgbd/rgbd_dataset_freiburg1_xyz}"
OUTPUT_DIR="${2:-${ROOT_DIR}/outputs/tum_freiburg1_xyz}"
MAX_FRAMES="${3:-0}"

# On WSL, prefer NVIDIA's driver-provided CUDA loader path.
if [[ -d "/usr/lib/wsl/lib" ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Dataset not found at ${DATASET_ROOT}; downloading default TUM sequence..."
  "${ROOT_DIR}/scripts/download_tum_rgbd.sh" rgbd_dataset_freiburg1_xyz "${ROOT_DIR}/third_party_datasets/tum_rgbd"
fi

mkdir -p "${OUTPUT_DIR}" "${ROOT_DIR}/.tools/rerun"

if [[ ! -x "${ROOT_DIR}/.tools/rerun/rerun" ]]; then
  echo "Rerun CLI not found; installing local copy..."
  "${ROOT_DIR}/scripts/install_rerun_cli.sh" "${ROOT_DIR}/.tools/rerun"
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DCUVSLAM_ENABLE_RERUN=ON
cmake --build "${BUILD_DIR}" -j"$(nproc)"

RERUN_BIN="${ROOT_DIR}/.tools/rerun/rerun"
export PATH="${ROOT_DIR}/.tools/rerun:${PATH}"

CMD=("${BUILD_DIR}/cuvslam_cli"
  --dataset_root "${DATASET_ROOT}"
  --dataset_format tum
  --output_dir "${OUTPUT_DIR}"
  --reference_tum "${DATASET_ROOT}/groundtruth.txt"
  --enable_rerun
  --rerun_save "${OUTPUT_DIR}/cuvslam.rrd"
  --rerun_log_every_n 5
)

if [[ "${MAX_FRAMES}" != "0" ]]; then
  CMD+=(--max_frames "${MAX_FRAMES}")
fi

if [[ "${4:-}" == "--spawn" ]]; then
  CMD+=(--rerun_spawn)
fi

"${CMD[@]}"

echo ""
echo "Performance report: ${OUTPUT_DIR}/performance_report.md"
sed -n '1,180p' "${OUTPUT_DIR}/performance_report.md"
echo ""
echo "Rerun file saved: ${OUTPUT_DIR}/cuvslam.rrd"
echo "Open with: ${RERUN_BIN} ${OUTPUT_DIR}/cuvslam.rrd"
