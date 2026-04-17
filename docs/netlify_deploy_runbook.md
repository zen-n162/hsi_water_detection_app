# Netlify Deploy Runbook

This runbook defines the exact Netlify setup for the production frontend.

## Service type

- Netlify static site

## GitHub connection

1. In Netlify, choose `Add new project`
2. Connect GitHub
3. Select the repository that contains `hsi_water_detection_app`
4. Choose the release branch

## Monorepo values

Use these exact values:

- Base directory: `frontend`
- Package directory: leave blank
- Build command: `npm run build`
- Publish directory: `dist`

Reasoning:

- the frontend package lives in `frontend/`
- [frontend/netlify.toml](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/netlify.toml) expects build execution inside that directory

## Environment variables

Set these in:

- Netlify dashboard
- Site configuration
- Environment variables

Production values:

```text
VITE_API_BASE_URL=https://<your-render-service>.onrender.com
VITE_APP_MODE=public
VITE_DEFAULT_DEVICE=cpu
VITE_ALLOWED_DEVICES=cpu
VITE_DEFAULT_MODEL_TYPE=ss
VITE_DEFAULT_SENSOR=hyperion
VITE_SHOW_SERVER_PATH_INPUTS=false
VITE_SHOW_DEPLOY_CONFIG_INPUT=false
VITE_SHOW_ADVANCED_OVERRIDES=false
```

## Preview deploy vs production deploy

### Preview deploy

Recommended approach:

- use the same frontend build variables unless you also provision a preview backend
- if preview deploys should hit production Render, state that explicitly in the release record

If you later add a preview Render backend, only these values change:

- `VITE_API_BASE_URL`

### Production deploy

- must point to the production Render backend
- must use `VITE_APP_MODE=public`

## SPA routing

Current status:

- no React Router is used
- no catch-all redirect is required today

If routes are later added, create a `_redirects` file or Netlify redirect rule at that time.

## Render URL change procedure

If the Render backend URL changes:

1. update `VITE_API_BASE_URL` in Netlify
2. trigger a new Netlify deploy
3. verify:
   - frontend root loads
   - `/health` is reachable via the new backend URL
   - preview works
   - inference works

## First production deploy checklist

1. confirm the Render backend is already live
2. confirm `/health` returns `ok: true`
3. set the Netlify environment variables
4. deploy the frontend
5. open the site and verify the header `HSI Water Detection UI`
6. confirm the `Resolved inference settings` panel loads deploy info from the backend

## Source of truth files

- [frontend/netlify.toml](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/netlify.toml)
- [frontend/.env.production.example](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/.env.production.example)
- [docs/production_env_matrix.md](/home/zennakamura/MasterResearch/hsi_water_detection_app/docs/production_env_matrix.md)
