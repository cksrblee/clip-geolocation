#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL="ViT-L-14"
PYTHON_BIN="python3"
SKIP_SPLITS=0
SKIP_NONGEO=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --skip-splits) SKIP_SPLITS=1; shift ;;
        --skip-nongeo) SKIP_NONGEO=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

mkdir -p logs results checkpoints interventions/val
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

run_step() {
    local name="$1"
    shift
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $name"
    "$@" 2>&1 | tee "logs/${name}.log"
}

if [[ $SKIP_SPLITS -eq 0 ]]; then
    run_step "00_build_splits" "$PYTHON_BIN" scripts/00_build_splits.py
fi
run_step "01_train_baselines" "$PYTHON_BIN" scripts/01_train_baselines.py --model "$MODEL"
run_step "02_build_interventions" "$PYTHON_BIN" scripts/02_build_interventions.py
run_step "03_audit_interventions" "$PYTHON_BIN" scripts/03_audit_interventions.py
run_step "04_evaluate_bundles" "$PYTHON_BIN" scripts/04_evaluate_bundles.py --model "$MODEL"
run_step "05_compute_visual_cue_statistics" "$PYTHON_BIN" scripts/05_compute_visual_cue_statistics.py
run_step "06_prompt_grounding" "$PYTHON_BIN" scripts/06_prompt_grounding.py --model "$MODEL"
if [[ $SKIP_NONGEO -eq 0 ]]; then
    run_step "07_nongeo_control" "$PYTHON_BIN" scripts/07_nongeo_control.py --model "$MODEL"
fi

echo "Experiment pipeline completed. Results: $REPO_ROOT/results"
