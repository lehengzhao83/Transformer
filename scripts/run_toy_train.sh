#!/usr/bin/env bash
set -Eeuo pipefail

# repo root = parent of this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEVICE="${DEVICE:-cpu}"

echo "[Info] Repo root: ${REPO_ROOT}"
echo "[Info] Using device: ${DEVICE}"

python -m transformer.train --device "${DEVICE}" --task copy --epochs 10
python -m transformer.train --device "${DEVICE}" --task reverse --epochs 20
