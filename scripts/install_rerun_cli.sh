#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${1:-${ROOT_DIR}/.tools/rerun}"
TARGET="${INSTALL_DIR}/rerun"

mkdir -p "${INSTALL_DIR}"

URL="$(python3 - <<'PY'
import json
import urllib.request

api='https://api.github.com/repos/rerun-io/rerun/releases/latest'
with urllib.request.urlopen(api) as r:
    data=json.load(r)

asset=None
for a in data.get('assets', []):
    n=a.get('name','')
    if n == 'rerun-cli-' + data['tag_name'].lstrip('v') + '-x86_64-unknown-linux-gnu':
        asset=a
        break

if asset is None:
    for a in data.get('assets', []):
        n=a.get('name','')
        if n.startswith('rerun-cli-') and n.endswith('-x86_64-unknown-linux-gnu'):
            asset=a
            break

if asset is None:
    raise SystemExit('Could not find rerun-cli Linux x86_64 asset in latest release')

print(asset['browser_download_url'])
PY
)"

echo "Downloading Rerun CLI from ${URL}"
curl -L --fail --retry 3 --retry-delay 2 -o "${TARGET}" "${URL}"
chmod +x "${TARGET}"

echo "Installed: ${TARGET}"
"${TARGET}" --version || true
