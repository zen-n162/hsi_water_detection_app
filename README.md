# hsi_water_detection_app

Research and demo application for hyperspectral water detection built around a frozen HyperSIGMA deployment profile.

This repository contains two intentionally separate layers:

1. a research pipeline for dataset building, training, evaluation, calibration, and freeze artifacts
2. a web-facing frontend/backend demo stack for ROI preview and inference with provenance

## Recommended deployment

The prepared public topology is:

- Frontend: Netlify
- Backend API: Render
- Runtime assets: mounted under `runtime/` on the Render service

The codebase now separates:

- research-local assets and workflows
- upload-first public web behavior
- environment-based deployment settings

See:

- [Web deployment audit](docs/web_deployment_audit.md)
- [Web deployment overview](docs/web_deployment.md)
- [Netlify + Render deploy guide](docs/netlify_render_deploy_guide.md)
- [Web release checklist](docs/web_release_checklist.md)

## Current frozen profile

Local research/default profile:

- Config: `configs/deploy/hypersigma_v3_calibrated.json`
- Run: `E02_head_only_posw_v3_block224`
- Model type: `ss`
- Threshold: `0.327428693347738`

Public web profile:

- Config: `configs/deploy/hypersigma_v3_calibrated_web.json`
- Output root: `runtime/outputs`
- Mounted checkpoint root: `runtime/assets/models/...`

## Runtime modes

See [Runtime modes](docs/runtime_modes.md).

The short version:

- `local-research`: research training/eval and freeze workflows
- `local-web-dev`: local frontend/backend development with optional server-side paths
- `public-web`: upload-first public demo behavior with path overrides disabled

## Local development

### Backend

```bash
cd /home/zennakamura/MasterResearch/hsi_water_detection_app
cp backend/.env.example backend/.env
PYTHONPATH=src:. uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

If you need the legacy conda subprocess behavior, set:

```bash
export HSI_INFERENCE_RUNTIME=conda
export HSI_INFERENCE_CONDA_ENV=HyperSIGMA
```

### Frontend

```bash
cd /home/zennakamura/MasterResearch/hsi_water_detection_app/frontend
cp .env.example .env.local
npm install
npm run dev -- --host
```

## Public deployment files added

- `frontend/.env.example`
- `frontend/.env.production.example`
- `frontend/netlify.toml`
- `backend/.env.example`
- `render.yaml`
- `configs/deploy/hypersigma_v3_calibrated_web.json`

## Important constraints

- The backend can now boot without immediately resolving the external HyperSIGMA repository, but real inference still requires mounted runtime model assets.
- The default public mode disables server-side file paths and deploy-config overrides.
- The current HyperSIGMA public demo path is still CPU/GPU environment sensitive because it depends on external upstream weights and a large fine-tuned checkpoint.

## Core documents

- [GUI inference contract](docs/gui_inference_contract.md)
- [Runtime modes](docs/runtime_modes.md)
- [Research pipeline](docs/research_pipeline.md)
- [Model card: HyperSIGMA v3 calibrated](docs/model_card_hypersigma_v3_calibrated.md)
