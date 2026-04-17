# Netlify Runbook For Local GPU Backend

This runbook assumes:

- frontend on Netlify Free
- backend on the research PC
- backend URL exposed through Cloudflare Tunnel or Tailscale Funnel

## GitHub connection

1. In Netlify, choose `Add new project`.
2. Connect GitHub.
3. Select the repository containing `hsi_water_detection_app`.
4. Choose the release branch.

## Build settings

Use these exact values:

- Base directory: `frontend`
- Package directory: leave blank
- Build command: `npm run build`
- Publish directory: `dist`

## Frontend environment variables

Use [frontend/.env.local_gpu_web.example](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/.env.local_gpu_web.example) as the source of truth.

Production values:

```text
VITE_API_BASE_URL=https://<your-tunnel-domain>
VITE_APP_MODE=local_gpu_web
VITE_DEFAULT_DEVICE=cuda
VITE_ALLOWED_DEVICES=cuda
VITE_DEFAULT_MODEL_TYPE=ss
VITE_DEFAULT_SENSOR=hyperion
VITE_SHOW_SERVER_PATH_INPUTS=false
VITE_SHOW_DEPLOY_CONFIG_INPUT=false
VITE_SHOW_ADVANCED_OVERRIDES=false
```

## Deploy order

1. start the backend locally
2. start the tunnel
3. confirm `https://<your-tunnel-domain>/health`
4. update Netlify env vars
5. trigger the Netlify deploy
6. run the live acceptance script

## If the tunnel URL changes

Update all of these together:

- Netlify `VITE_API_BASE_URL`
- backend `HSI_PUBLIC_BASE_URL`
- backend `HSI_CORS_ALLOW_ORIGINS` if the frontend hostname also changed

Then redeploy the frontend and restart the backend if `HSI_PUBLIC_BASE_URL` changed.

## Preview deploys

For preview deploys, choose one policy and document it:

1. disable preview deploy testing against the live backend
2. allow preview deploys but point them at the production backend
3. provision a separate tunnel URL just for preview

For a single-owner research demo, option `2` is the simplest but increases accidental public testing against the live GPU backend.

## Smoke test

After a production deploy:

1. open the Netlify site
2. confirm the owner-operated GPU banner is visible
3. confirm the `Resolved inference settings` panel loads
4. run preview
5. run inference
6. confirm no local absolute path is shown in the UI
