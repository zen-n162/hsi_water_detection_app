#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-HyperSIGMA}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-$ROOT_DIR/configs/deploy/hypersigma_v3_calibrated.json}"
INPUT_PATH="${INPUT_PATH:-/home/zennakamura/MasterResearch/HyperSIGMA/HyperspectralDetection/Hyperion_WaterLabel_20111222/data/processed/hyperion/hyperion_stack_crop_f16.tif}"
ROW_START="${ROW_START:-128}"
ROW_STOP="${ROW_STOP:-192}"
COL_START="${COL_START:-2368}"
COL_STOP="${COL_STOP:-2432}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found."
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' was not found."
  exit 1
fi

echo "Release candidate reproduction"
echo "root_dir=$ROOT_DIR"
echo "deploy_config=$DEPLOY_CONFIG"
echo "env_name=$ENV_NAME"

echo "[1/4] Building frontend"
npm --prefix "$ROOT_DIR/frontend" ci
npm --prefix "$ROOT_DIR/frontend" run build

echo "[2/4] Deploy config"
echo "$DEPLOY_CONFIG"

echo "[3/4] Backend startup command"
echo "PYTHONPATH=src:. MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/xdg-cache uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"

echo "[4/4] GUI smoke test command"
echo "PYTHONPATH=src:. conda run -n $ENV_NAME python research/evaluation/smoke_test_gui_pipeline.py --deploy_config $DEPLOY_CONFIG --input_path $INPUT_PATH --sensor hyperion --device cuda --row_start $ROW_START --row_stop $ROW_STOP --col_start $COL_START --col_stop $COL_STOP --output_json experiments/eval_v3/gui_smoke_test_result.json"
