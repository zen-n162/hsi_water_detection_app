# Local GPU Web Rollback Plan

## 1. Immediate stop

If the public demo must be stopped quickly:

1. stop the tunnel first
2. confirm public `/health` is unreachable
3. leave Netlify up or switch it to a maintenance message if needed

## 2. Tunnel rollback

Cloudflare Tunnel:

- stop `cloudflared` or disable the OS service
- if the hostname changed, restore the previous tunnel or previous public hostname mapping

Tailscale Funnel:

- run `tailscale funnel --https=443 off` or `tailscale funnel reset`
- confirm the public `*.ts.net` URL no longer serves the backend

## 3. Frontend rollback

If Netlify points at the wrong backend URL:

1. restore the previous `VITE_API_BASE_URL`
2. trigger a new Netlify deploy or restore the last known-good deploy
3. reopen the frontend and confirm it points at the intended backend

## 4. Backend rollback

If the local backend code or deploy config is the problem:

1. stop the local uvicorn process
2. switch back to the last known-good commit or branch
3. restore the previous deploy config if it changed
4. restart the backend locally
5. retest local `/health` before bringing the tunnel back

## 5. Model config rollback

If the issue comes from model/calibration/provenance settings:

1. restore the previous [configs/deploy/hypersigma_v3_local_gpu_web.json](/home/zennakamura/MasterResearch/hsi_water_detection_app/configs/deploy/hypersigma_v3_local_gpu_web.json)
2. restart the backend
3. rerun preview and inference locally
4. only then reopen the tunnel

## 6. URL rollback sequence

When the tunnel URL changes unexpectedly:

1. decide the old stable URL or new stable URL
2. set backend `HSI_PUBLIC_BASE_URL` to that URL
3. set Netlify `VITE_API_BASE_URL` to the same URL
4. redeploy frontend
5. rerun acceptance

## 7. Files and values to inspect first

- [backend/.env.local_gpu_web.example](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/.env.local_gpu_web.example)
- [frontend/.env.local_gpu_web.example](/home/zennakamura/MasterResearch/hsi_water_detection_app/frontend/.env.local_gpu_web.example)
- [configs/deploy/hypersigma_v3_local_gpu_web.json](/home/zennakamura/MasterResearch/hsi_water_detection_app/configs/deploy/hypersigma_v3_local_gpu_web.json)
- current tunnel URL
- current Netlify env `VITE_API_BASE_URL`
- current backend env `HSI_PUBLIC_BASE_URL`

## 8. Recovery validation after rollback

1. local `/health`
2. public `/health`
3. one preview request
4. one inference request
5. `metadata.json`
