#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PYTHONPATH:-src:.}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"

export HSI_APP_MODE="${HSI_APP_MODE:-local}"
export HSI_DEPLOY_CONFIG="${HSI_DEPLOY_CONFIG:-configs/deploy/hypersigma_confmask_thr008_splitquality_stress_seed13.json}"
export HSI_OUTPUT_ROOT="${HSI_OUTPUT_ROOT:-outputs}"
export HSI_OUTPUT_URL_PREFIX="${HSI_OUTPUT_URL_PREFIX:-/outputs}"
export HSI_CORS_ALLOW_ORIGINS="${HSI_CORS_ALLOW_ORIGINS:-http://127.0.0.1:5173,http://localhost:5173}"
export HSI_ALLOW_SERVER_FILE_PATHS="${HSI_ALLOW_SERVER_FILE_PATHS:-true}"
export HSI_ALLOW_DEPLOY_CONFIG_OVERRIDE="${HSI_ALLOW_DEPLOY_CONFIG_OVERRIDE:-true}"

export HSI_DEFAULT_DEVICE="${HSI_DEFAULT_DEVICE:-cuda}"
export HSI_INFERENCE_RUNTIME="${HSI_INFERENCE_RUNTIME:-conda}"
export HSI_INFERENCE_CONDA_ENV="${HSI_INFERENCE_CONDA_ENV:-HyperSIGMA}"

HOST="${HSI_BACKEND_HOST:-127.0.0.1}"
PORT="${HSI_BACKEND_PORT:-8000}"

echo "Starting backend with deploy profile: ${HSI_DEPLOY_CONFIG}"
echo "Inference runtime: ${HSI_INFERENCE_RUNTIME} (${HSI_INFERENCE_CONDA_ENV})"
exec uvicorn backend.app.main:app --host "${HOST}" --port "${PORT}"
