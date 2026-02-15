#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEQUENCE="${1:-rgbd_dataset_freiburg1_xyz}"
DEST_ROOT="${2:-${ROOT_DIR}/third_party_datasets/tum_rgbd}"

case "${SEQUENCE}" in
  rgbd_dataset_freiburg1_*) GROUP="freiburg1" ;;
  rgbd_dataset_freiburg2_*) GROUP="freiburg2" ;;
  rgbd_dataset_freiburg3_*) GROUP="freiburg3" ;;
  *)
    echo "Unsupported sequence '${SEQUENCE}'. Expected a TUM RGB-D sequence starting with rgbd_dataset_freiburg{1|2|3}_" >&2
    exit 2
    ;;
esac

ARCHIVE_DIR="${DEST_ROOT}/archives"
DATASET_DIR="${DEST_ROOT}/${SEQUENCE}"
ARCHIVE_PATH="${ARCHIVE_DIR}/${SEQUENCE}.tgz"
URL="https://cvg.cit.tum.de/rgbd/dataset/${GROUP}/${SEQUENCE}.tgz"

mkdir -p "${ARCHIVE_DIR}" "${DEST_ROOT}"

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  echo "Downloading ${URL}"
  curl -L --fail --retry 3 --retry-delay 2 -o "${ARCHIVE_PATH}" "${URL}"
else
  echo "Using existing archive: ${ARCHIVE_PATH}"
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Extracting to ${DEST_ROOT}"
  tar -xzf "${ARCHIVE_PATH}" -C "${DEST_ROOT}"
else
  echo "Dataset already extracted: ${DATASET_DIR}"
fi

echo "Dataset ready: ${DATASET_DIR}"
echo "Ground truth file: ${DATASET_DIR}/groundtruth.txt"
