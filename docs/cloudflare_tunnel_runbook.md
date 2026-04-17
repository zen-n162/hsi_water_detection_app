# Cloudflare Tunnel Runbook

This is the primary exposure method for `local_gpu_web`.

## Why Cloudflare Tunnel is the primary choice

- stable custom hostname support
- outbound-only connector model
- good fit for a Netlify frontend calling a single backend origin
- optional Cloudflare Access layer if you later want login-gated access

Cloudflare currently recommends remotely-managed tunnels for most use cases, while locally-managed tunnels remain available for development, testing, or legacy workflows.

## Recommended setup

Use a remotely-managed tunnel with a stable hostname such as:

- `https://hsi-api.<your-domain>`

Point that hostname at the FastAPI server running on:

- `http://127.0.0.1:8000`

## Before you start

1. Copy [backend/.env.local_gpu_web.example](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/.env.local_gpu_web.example) to `backend/.env.local`.
2. Set:
   - `HSI_CORS_ALLOW_ORIGINS=https://<your-netlify-site>.netlify.app`
   - `HSI_PUBLIC_BASE_URL=https://hsi-api.<your-domain>`
3. Confirm the deploy config exists:
   - [configs/deploy/hypersigma_v3_local_gpu_web.json](/home/zennakamura/MasterResearch/hsi_water_detection_app/configs/deploy/hypersigma_v3_local_gpu_web.json)
4. Confirm the local GPU environment can import and run HyperSIGMA.

## Start the backend first

Run the backend on the research PC:

```bash
cd /home/zennakamura/MasterResearch/hsi_water_detection_app
PYTHONPATH=src:. uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

If you want persistent logs for acceptance and incident review:

```bash
cd /home/zennakamura/MasterResearch/hsi_water_detection_app
mkdir -p outputs/logs
PYTHONPATH=src:. uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 \
  2>&1 | tee outputs/logs/local_gpu_web_backend.log
```

Check locally:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/inference/deploy-config
```

## Create the Cloudflare tunnel

Dashboard flow:

1. Log in to Cloudflare One.
2. Go to `Networks > Connectors > Cloudflare Tunnels`.
3. Select `Create a tunnel`.
4. Choose `Cloudflared`.
5. Name the tunnel, for example `hsi-local-gpu`.
6. Add a public hostname such as `hsi-api.<your-domain>`.
7. Set the service type to `HTTP`.
8. Set the origin service URL to `http://127.0.0.1:8000`.

## Run the connector on the PC

Install `cloudflared`, then run the tunnel with the token copied from the dashboard:

```bash
cloudflared tunnel run --token <TUNNEL_TOKEN>
```

If you want the tunnel to survive reboots, install it as an OS service and put the token into the service environment instead of leaving it in shell history.

## Optional locally-managed fallback

If you prefer CLI-managed config on the PC:

```bash
cloudflared tunnel login
cloudflared tunnel create hsi-local-gpu
cloudflared tunnel route dns hsi-local-gpu hsi-api.<your-domain>
```

Create a config file like:

```yaml
tunnel: <TUNNEL_UUID>
credentials-file: /home/<user>/.cloudflared/<TUNNEL_UUID>.json
ingress:
  - hostname: hsi-api.<your-domain>
    service: http://127.0.0.1:8000
  - service: http_status:404
```

Validate it:

```bash
cloudflared tunnel ingress validate
cloudflared tunnel ingress rule https://hsi-api.<your-domain>
cloudflared tunnel run hsi-local-gpu
```

## Netlify values that must match

- `VITE_API_BASE_URL=https://hsi-api.<your-domain>`
- backend `HSI_PUBLIC_BASE_URL=https://hsi-api.<your-domain>`
- backend `HSI_CORS_ALLOW_ORIGINS=https://<your-netlify-site>.netlify.app`

## Smoke test after tunnel comes up

```bash
curl https://hsi-api.<your-domain>/health
curl https://hsi-api.<your-domain>/inference/deploy-config
```

Then run:

```bash
cd /home/zennakamura/MasterResearch/hsi_water_detection_app
python research/evaluation/live_web_acceptance_checklist_local_gpu.py \
  --frontend-url https://<your-netlify-site>.netlify.app \
  --backend-url https://hsi-api.<your-domain> \
  --sample-hsi /absolute/path/to/hyperion_stack_crop_f16.tif \
  --sensor hyperion \
  --device cuda \
  --backend-log-file outputs/logs/local_gpu_web_backend.log
```

## Security notes

- Do not expose `HSI_ALLOW_SERVER_FILE_PATHS=true`.
- Do not expose `HSI_ALLOW_DEPLOY_CONFIG_OVERRIDE=true`.
- Treat the tunnel token as a secret. Anyone with the token can run the tunnel.
- Prefer a dedicated hostname for the API.
- Consider Cloudflare Access if the demo should not be fully public.

## Stop public access

Foreground process:

- `Ctrl+C` in the `cloudflared` terminal

Service-managed connector:

- stop the `cloudflared` service from the OS service manager

Then verify the public hostname no longer reaches `/health`.

## Recovery after PC reboot

1. Confirm the GPU runtime is healthy.
2. Start FastAPI and confirm local `/health`.
3. Start `cloudflared` or confirm the service auto-started.
4. Re-check the public `/health`.
5. Re-run the acceptance script if the outage affected users.
