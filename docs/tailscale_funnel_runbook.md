# Tailscale Funnel Runbook

This is the secondary exposure method for `local_gpu_web`.

## When to choose Tailscale Funnel

Use Tailscale Funnel when:

- you do not want to manage a Cloudflare-hosted domain
- you already run Tailscale on the research PC
- you are comfortable with a `*.ts.net` public URL

Tradeoffs:

- Funnel is still documented as beta
- URL branding is less flexible than Cloudflare custom hostnames
- feature enablement happens through Tailscale policy and HTTPS certificate setup

## Prepare the backend

Copy [backend/.env.local_gpu_web.example](/home/zennakamura/MasterResearch/hsi_water_detection_app/backend/.env.local_gpu_web.example) to `backend/.env.local` and set:

- `HSI_CORS_ALLOW_ORIGINS=https://<your-netlify-site>.netlify.app`
- `HSI_PUBLIC_BASE_URL=https://<your-node>.<your-tailnet>.ts.net`

Start FastAPI locally:

```bash
cd /home/zennakamura/MasterResearch/hsi_water_detection_app
PYTHONPATH=src:. uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

## Enable Funnel

1. Install and sign in to Tailscale on the research PC.
2. Make sure HTTPS certificates are enabled for the tailnet.
3. Run:

```bash
tailscale funnel --bg 127.0.0.1:8000
```

Check status:

```bash
tailscale funnel status
tailscale funnel status --json
```

The CLI prints the public `https://<node>.<tailnet>.ts.net` URL. Use that as:

- frontend `VITE_API_BASE_URL`
- backend `HSI_PUBLIC_BASE_URL`

## Netlify values that must match

- `VITE_API_BASE_URL=https://<node>.<tailnet>.ts.net`
- `VITE_APP_MODE=local_gpu_web`
- backend `HSI_PUBLIC_BASE_URL=https://<node>.<tailnet>.ts.net`
- backend `HSI_CORS_ALLOW_ORIGINS=https://<your-netlify-site>.netlify.app`

## Stop public access

To turn off the specific Funnel:

```bash
tailscale funnel --https=443 off
```

To clear Funnel configuration:

```bash
tailscale funnel reset
```

## Recovery after reboot

If Funnel was started with `--bg`, Tailscale documents that it resumes after reboot or after `tailscale down` / `tailscale up`.

Recommended post-reboot sequence:

1. confirm `tailscale status`
2. start FastAPI if it did not auto-start
3. confirm `tailscale funnel status`
4. verify public `/health`

## Security notes

- Funnel must be enabled in the tailnet policy for the node.
- Keep the backend bound to `127.0.0.1`.
- Do not rely on Funnel for app-level authentication. Add another control if the demo should not be internet-wide.
- Keep override envs disabled exactly as in `local_gpu_web`.
