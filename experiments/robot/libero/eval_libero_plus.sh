#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <TASK_SUITE|summary> <SAVE_NAME> [--pretrained_checkpoint PATH] [--resume] [extra args...]"
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
MODE_OR_SUITE=$1
SAVE_NAME=$2
shift 2

if [[ "${MODE_OR_SUITE}" == "summary" ]]; then
  python "${SCRIPT_DIR}/run_libero_plus_eval.py" \
    --summary_only --save_name "${SAVE_NAME}" "$@"
else
  SUITE=${MODE_OR_SUITE#libero_}
  python "${SCRIPT_DIR}/run_libero_plus_eval.py" \
    --task_suite_name "libero_${SUITE}" --save_name "${SAVE_NAME}" "$@"
fi
