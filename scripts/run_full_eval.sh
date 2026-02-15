#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
DATASET_ROOT="${1:-${ROOT_DIR}/data_sample}"
OUTPUT_DIR="${2:-${ROOT_DIR}/outputs/full_run}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"$(nproc)"

"${BUILD_DIR}/cuvslam_cli" \
  --dataset_root "${DATASET_ROOT}" \
  --output_dir "${OUTPUT_DIR}"

echo ""
echo "Generated report: ${OUTPUT_DIR}/performance_report.md"
sed -n '1,160p' "${OUTPUT_DIR}/performance_report.md"
