# Local GPU Web Architecture

## Goal

Expose the existing HyperSIGMA GUI as a low-cost public demo without moving GPU inference off the research PC.

## Adopted topology

- Frontend: Netlify Free
- Backend: FastAPI running on the owner-operated research PC
- Inference device: local GPU (`cuda`)
- Primary exposure: Cloudflare Tunnel
- Secondary exposure: Tailscale Funnel
- Runtime mode: `local_gpu_web`

## Why this architecture

- Netlify keeps the frontend cheap and globally reachable.
- The research PC keeps the GPU, local checkpoint, calibration JSON, manifest, dataset, and upstream HyperSIGMA assets in place.
- The tunnel removes the need to open inbound router ports.
- `local_gpu_web` keeps the public GUI upload-first while locking down unsafe overrides.

## Trust boundary

Public users can:

- open the Netlify frontend
- upload HSI input files
- request grayscale preview and inference
- read public-safe metadata and result images

Public users cannot:

- submit server-side local paths
- switch deploy config files
- override checkpoint, calibration JSON, manifest, dataset, patch size, or stride
- read internal absolute paths from API responses or `metadata.json`

## Runtime behavior in `local_gpu_web`

- Backend resolves the fixed deploy profile from [configs/deploy/hypersigma_v3_local_gpu_web.json](/home/zennakamura/MasterResearch/hsi_water_detection_app/configs/deploy/hypersigma_v3_local_gpu_web.json).
- Backend defaults the device to `cuda`.
- Preview and inference outputs are served from `/outputs/...`.
- Public responses expose only safe provenance such as deploy name, run name, threshold, model type, and executed device.
- Internal absolute paths remain visible only in backend-side logs.

## Core environment variables

Backend:

- `HSI_APP_MODE=local_gpu_web`
- `HSI_DEPLOY_CONFIG=configs/deploy/hypersigma_v3_local_gpu_web.json`
- `HSI_CORS_ALLOW_ORIGINS=https://<your-netlify-site>.netlify.app`
- `HSI_OUTPUT_ROOT=outputs/web_ui`
- `HSI_DEFAULT_DEVICE=cuda`
- `HSI_PUBLIC_BASE_URL=https://<your-tunnel-domain>`

Frontend:

- `VITE_API_BASE_URL=https://<your-tunnel-domain>`
- `VITE_APP_MODE=local_gpu_web`
- `VITE_DEFAULT_DEVICE=cuda`
- `VITE_ALLOWED_DEVICES=cuda`

## Operational notes

- Run FastAPI on `127.0.0.1:8000` and let the tunnel proxy that local service.
- Keep the tunnel URL stable. If it changes, update both Netlify and `HSI_PUBLIC_BASE_URL`.
- If the PC sleeps, reboots, loses network, or the GPU runtime is unavailable, the public demo will fail even if Netlify stays online.
