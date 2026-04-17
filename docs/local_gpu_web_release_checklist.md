# Local GPU Web Release Checklist

This checklist is intentionally ordered for the `Netlify + local GPU backend + tunnel` release path.

## 1. Freeze and local runtime sanity

- [ ] `git status` has been reviewed and the release scope is understood
- [ ] the active demo deploy profile is [configs/deploy/hypersigma_v3_local_gpu_web.json](/home/zennakamura/MasterResearch/hsi_water_detection_app/configs/deploy/hypersigma_v3_local_gpu_web.json)
- [ ] local GPU runtime is healthy
- [ ] HyperSIGMA local assets still exist at the configured paths

## 2. Backend env setup

- [ ] `backend/.env.local` exists and follows [backend/.env.local_gpu_web.example](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/.env.local_gpu_web.example)
- [ ] `HSI_APP_MODE=local_gpu_web`
- [ ] `HSI_DEFAULT_DEVICE=cuda`
- [ ] `HSI_ALLOW_SERVER_FILE_PATHS=false`
- [ ] `HSI_ALLOW_DEPLOY_CONFIG_OVERRIDE=false`

## 3. Frontend env setup

- [ ] Netlify env vars match [frontend/.env.local_gpu_web.example](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/.env.local_gpu_web.example)
- [ ] `VITE_API_BASE_URL` matches the tunnel URL
- [ ] `VITE_APP_MODE=local_gpu_web`
- [ ] `VITE_ALLOWED_DEVICES=cuda`

## 4. Local backend boot

- [ ] `PYTHONPATH=src:. uvicorn backend.app.main:app --host 127.0.0.1 --port 8000` starts successfully
- [ ] local `/health` returns `app_mode=local_gpu_web`
- [ ] local `/inference/deploy-config` returns `provenance_visibility=public_safe`

## 5. Tunnel boot

- [ ] Cloudflare Tunnel or Tailscale Funnel is running
- [ ] the public tunnel URL is fixed
- [ ] public `/health` works through the tunnel

## 6. Netlify deploy

- [ ] Netlify env vars are updated to the current tunnel URL
- [ ] frontend deploy is triggered
- [ ] the Netlify production site loads

## 7. Live checks

- [ ] frontend root loads
- [ ] preview works
- [ ] ROI selection works
- [ ] inference works
- [ ] Pseudo Color, Overlay, Spatial, and Spectral images render
- [ ] `metadata.json` is public-safe
- [ ] `executed_device` is `cuda`

## 8. Acceptance script

- [ ] `research/evaluation/live_web_acceptance_checklist_local_gpu.py` exits with code `0`
- [ ] optional backend log verification passes if a log file is supplied

## 9. Release record

- [ ] tunnel URL is recorded
- [ ] Netlify deploy ID is recorded
- [ ] backend commit SHA/tag is recorded
- [ ] rollback steps from [docs/local_gpu_web_rollback_plan.md](/home/zennakamura/MasterResearch/hsi_water_detection_app/docs/local_gpu_web_rollback_plan.md) are attached

## 10. Public stop procedure is known

- [ ] you know how to stop the tunnel immediately
- [ ] you know how to redeploy Netlify to the previous API URL
- [ ] you know how to stop the backend on the research PC
