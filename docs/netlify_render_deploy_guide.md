# Netlify + Render Deploy Guide

This guide prepares the current repository for a public demo without changing the frozen research defaults.

## 1. Prepare the repository

From the project root:

```bash
cd /home/zennakamura/MasterResearch/hsi_water_detection_app
```

Confirm these files exist:

- `frontend/netlify.toml`
- `frontend/.env.production.example`
- `backend/.env.example`
- `render.yaml`
- `configs/deploy/hypersigma_v3_calibrated_web.json`

## 2. Prepare Render runtime assets

Upload or copy these into the Render persistent disk layout:

```text
runtime/
  assets/
    models/
      E02_head_only_posw_v3_block224/
        model_best.pt
  upstream/
    HyperSIGMA/
      HyperspectralDetection/
        spat-vit-b-checkpoint-1599.pth
        spec-vit-b-checkpoint-1599.pth
        Target_Detection/
```

## 3. Create the Render backend

Use the repository root `render.yaml`.

Important runtime settings:

- service type: web
- health check path: `/health`
- start command: `PYTHONPATH=src:. uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
- persistent disk mount path: `/opt/render/project/src/runtime`

Set `HSI_CORS_ALLOW_ORIGINS` to your Netlify production domain and any preview domains you intentionally allow.

## 4. Verify Render after first deploy

Check:

1. `/health` returns `ok: true`
2. `default_deploy_config_exists` is `true`
3. `output_root` points to `runtime/outputs`
4. the disk contains the required runtime assets

## 5. Create the Netlify frontend

Deploy the `frontend/` package.

Set these build variables:

```text
VITE_API_BASE_URL=https://<your-render-service>.onrender.com
VITE_APP_MODE=public
VITE_DEFAULT_DEVICE=cpu
VITE_ALLOWED_DEVICES=cpu
VITE_SHOW_SERVER_PATH_INPUTS=false
VITE_SHOW_DEPLOY_CONFIG_INPUT=false
VITE_SHOW_ADVANCED_OVERRIDES=false
```

`frontend/netlify.toml` already provides:

- build command: `npm run build`
- publish directory: `dist`
- Node version: `20`

## 6. End-to-end smoke test

Run this manual sequence after both services are live:

1. open the Netlify frontend
2. upload a sample HSI file
3. load grayscale preview
4. choose ROI
5. run inference
6. confirm result images load from the Render backend
7. open `metadata.json` and confirm provenance fields are present

## 7. Rollback guidance

If the public profile fails:

- keep the local profile untouched at `configs/deploy/hypersigma_v3_calibrated.json`
- switch the backend back to local-dev style env values
- do not overwrite or remove the frozen research artifacts

## Official references

- Netlify file-based configuration: https://docs.netlify.com/configure-builds/file-based-configuration/
- Netlify monorepo/package directory guidance: https://docs.netlify.com/configure-builds/monorepos/
- Netlify build environment variables: https://docs.netlify.com/build/configure-builds/environment-variables/
- Vite env variables: https://vite.dev/guide/env-and-mode
- Render FastAPI deploy guide: https://render.com/docs/deploy-fastapi
- Render Blueprint YAML reference: https://render.com/docs/blueprint-spec
- Render health checks: https://render.com/docs/health-checks
- Render persistent disks: https://render.com/docs/disks
