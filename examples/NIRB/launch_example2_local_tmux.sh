#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/examples/NIRB/logs"
mkdir -p "${LOG_DIR}"

SESSION_NAME="${1:-nirb_example2_local_fpi}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${SESSION_NAME}_${TIMESTAMP}.log"

cd "${ROOT_DIR}"

CMD=(
  "conda" "run" "--no-capture-output" "-n" "fenicsx"
  "python" "-u"
  "examples/NIRB/run_example2_local.py"
  "--output-dir" "examples/NIRB/artifacts/example2_local_fom_tmux"
  "--mesh-size" "0.04"
  "--mesh-order" "1"
  "--poly-order" "1"
  "--pressure-order" "1"
  "--dt" "0.008"
  "--end-time" "0.008"
  "--max-steps" "1"
  "--max-coupling-iters" "12"
  "--coupling-abs-tol" "1e-6"
  "--coupling-rel-tol" "1e-6"
  "--force-update" "constant"
  "--force-relaxation" "0.25"
  "--force-history" "50"
  "--force-regularization" "1e-10"
  "--newton-tol" "1e-6"
  "--max-newton-iter" "8"
  "--pressure-gauge" "1e-5"
  "--backend" "cpp"
  "--linear-backend" "petsc"
  "--snapshot-mode" "all"
  "--verbose"
)

tmux new-session -d -s "${SESSION_NAME}" "cd '${ROOT_DIR}' && ${CMD[*]} 2>&1 | tee '${LOG_FILE}'"

echo "session=${SESSION_NAME}"
echo "log=${LOG_FILE}"
echo "attach: tmux attach -t ${SESSION_NAME}"
