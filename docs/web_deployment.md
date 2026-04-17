# Web Deployment Overview

## Recommended topology

Use this split deployment for the first public release:

- Netlify serves the static React/Vite frontend
- Render serves the FastAPI backend and generated output images

Reasoning:

- the frontend is a plain static build
- the backend needs writable runtime storage for preview/inference artifacts
- heavy model assets should stay off the frontend and out of Git

## Public runtime design

### Code and config in Git

- `frontend/`
- `backend/`
- `src/`
- `configs/deploy/hypersigma_v3_calibrated_web.json`
- docs and freeze JSON

### Runtime assets outside Git

- fine-tuned checkpoint
- upstream HyperSIGMA repository
- upstream encoder checkpoints

### Runtime writable area

- preview PNGs
- inference overlays
- metadata JSON
- API result JSON

## Environment variables

### Frontend

Use these at build time:

- `VITE_API_BASE_URL`
- `VITE_APP_MODE`
- `VITE_DEFAULT_DEVICE`
- `VITE_ALLOWED_DEVICES`
- `VITE_SHOW_SERVER_PATH_INPUTS`
- `VITE_SHOW_DEPLOY_CONFIG_INPUT`
- `VITE_SHOW_ADVANCED_OVERRIDES`

### Backend

Use these at runtime:

- `HSI_APP_MODE`
- `HSI_DEFAULT_DEPLOY_CONFIG`
- `HSI_OUTPUT_ROOT`
- `HSI_OUTPUT_URL_PREFIX`
- `HSI_CORS_ALLOW_ORIGINS`
- `HSI_ALLOW_SERVER_FILE_PATHS`
- `HSI_ALLOW_DEPLOY_CONFIG_OVERRIDE`
- `HSI_INFERENCE_RUNTIME`
- `HYPERSIGMA_ROOT`
- `HYPERSIGMA_SPAT_CHECKPOINT`
- `HYPERSIGMA_SPEC_CHECKPOINT`

## Output serving strategy

The backend writes generated artifacts under the configured output root and mounts that directory as static files.

Expected public URL shape:

- `/outputs/preview_ui/<timestamp>/grayscale_preview.png`
- `/outputs/web_ui/<timestamp>/probability_overlay.png`

This keeps result hosting tied to the backend, which is the side that owns the writable filesystem.

## Local vs public differences

| Topic | Local web dev | Public web |
| --- | --- | --- |
| `input_path` | allowed | rejected |
| `deploy_config_path` override | allowed | rejected |
| output root | `outputs/` | `runtime/outputs` |
| frontend server path controls | visible | hidden by default |
| default device | usually `cuda` | usually `cpu` |

## Netlify note

The current app does not use React Router, so a catch-all SPA redirect is not required for the present route structure.

## Render note

The current public strategy assumes a persistent disk mounted at `runtime/` so that:

- heavy assets survive redeploys
- generated outputs survive restarts
- the codebase stays smaller and cleaner
