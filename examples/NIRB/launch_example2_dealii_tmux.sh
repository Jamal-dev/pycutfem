#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION_NAME="${1:-nirb_example2_smoke}"
shift || true

LOG_DIR="${ROOT_DIR}/examples/NIRB/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/${SESSION_NAME}_${STAMP}.log"
OUT_DIR="${ROOT_DIR}/examples/NIRB/artifacts/example2_local_smoke"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

EXTRA_ARGS=("$@")
CMD=(
  "cd" "${ROOT_DIR}"
  "&&"
  "conda" "run" "--no-capture-output" "-n" "fenicsx" "python" "-u"
  "examples/fsi_dealii_reference.py"
  "--mesh" "examples/meshes/fsi_conforming.msh"
  "--mesh-format" "gmsh"
  "--turek-case" "fsi2"
  "--n-steps" "1"
  "--output-dir" "examples/NIRB/artifacts/example2_local_smoke"
  "${EXTRA_ARGS[@]}"
  "2>&1" "|" "tee" "${LOG_PATH}"
)

tmux new-session -d -s "${SESSION_NAME}" "$(printf '%q ' "${CMD[@]}")"

echo "Started tmux session: ${SESSION_NAME}"
echo "Log file: ${LOG_PATH}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Tail log with: tail -f ${LOG_PATH}"
