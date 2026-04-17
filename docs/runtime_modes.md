# Runtime Modes

This repository now distinguishes three runtime modes so that research freeze work and public web deployment do not step on each other.

## 1. `local-research`

Purpose:

- training
- evaluation
- threshold selection
- calibration fitting
- release freeze reproduction

Characteristics:

- can reference local datasets and experiment directories
- can use ad-hoc research scripts
- can use GPU and local absolute paths
- should keep using `configs/deploy/hypersigma_v3_calibrated.json` for the frozen local default

## 2. `local-web-dev`

Purpose:

- local frontend/backend integration
- debugging upload flow, ROI UI, and output rendering
- validating deployment changes before cloud rollout

Recommended settings:

- `HSI_APP_MODE=local`
- `HSI_DEFAULT_DEPLOY_CONFIG=configs/deploy/hypersigma_v3_calibrated.json`
- `HSI_ALLOW_SERVER_FILE_PATHS=true`
- `HSI_ALLOW_DEPLOY_CONFIG_OVERRIDE=true`
- `VITE_APP_MODE=local`

Characteristics:

- frontend may show server-side path inputs
- backend may accept `input_path` and path overrides
- outputs are stored under `outputs/`

## 3. `public-web`

Purpose:

- Netlify + Render public demo
- upload-first inference flow
- minimal safe surface area

Recommended settings:

- `HSI_APP_MODE=public`
- `HSI_DEFAULT_DEPLOY_CONFIG=configs/deploy/hypersigma_v3_calibrated_web.json`
- `HSI_OUTPUT_ROOT=runtime/outputs`
- `HSI_ALLOW_SERVER_FILE_PATHS=false`
- `HSI_ALLOW_DEPLOY_CONFIG_OVERRIDE=false`
- `VITE_APP_MODE=public`

Characteristics:

- frontend hides local path and deploy-config override inputs
- backend rejects `input_path` and deploy-config override requests
- runtime outputs are served from the backend static mount at `/outputs/...`
- model checkpoints and upstream HyperSIGMA assets are expected outside Git, under `runtime/`

## GPU-dependent vs GPU-independent behavior

- Preview generation is GPU-independent.
- Backend boot and `/health` are GPU-independent.
- Real HyperSIGMA inference is environment-dependent and requires:
  - PyTorch + raster stack
  - upstream HyperSIGMA repository
  - encoder checkpoints
  - fine-tuned checkpoint
- Public demo should default the frontend device selector to `cpu` unless a GPU Render instance is intentionally provisioned.
