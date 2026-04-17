# Web Deployment Audit

Date: 2026-04-17

Scope:

- frontend static deployment readiness
- backend Render readiness
- public runtime asset strategy
- blockers for a stable upload-first public demo

## Executive summary

Netlify + Render is a workable first-choice deployment for this repository if the deployment is treated as:

- static frontend on Netlify
- upload-first API on Render
- heavy runtime artifacts mounted under `runtime/` on the Render service

The codebase was updated so that:

- frontend API base URL is environment-driven
- frontend public mode hides local path inputs by default
- backend CORS, output root, and deploy-config selection are environment-driven
- backend can boot before resolving the external HyperSIGMA repository
- public mode rejects server-side local path behavior

The main remaining operational risk is not code wiring. It is runtime asset placement:

- large fine-tuned checkpoint
- upstream HyperSIGMA repository
- upstream encoder checkpoints

## A. Frontend audit

| Item | Previous state | Audit result | Status |
| --- | --- | --- | --- |
| API base URL | hardcoded `http://127.0.0.1:18000` | moved to `VITE_API_BASE_URL`, with local default `http://127.0.0.1:8000` | fixed |
| Deploy config path | hardcoded absolute local path in `App.tsx` | frontend now asks backend for `/inference/deploy-config` | fixed |
| Public/local mode split | no split | `VITE_APP_MODE` and visibility flags added | fixed |
| Netlify config | absent | `frontend/netlify.toml` added | fixed |
| Static build viability | unclear | `npm run build` succeeds | fixed |
| React Router redirect need | none detected | no `react-router` usage found, no SPA redirect required for current app | no action needed |
| Production error handling | generic string only | structured API error parsing with hint/status added | fixed |

## B. Backend audit

| Item | Previous state | Audit result | Status |
| --- | --- | --- | --- |
| FastAPI entrypoint | `backend.app.main:app` exists | still valid as Render entrypoint | good |
| CORS | `allow_origins=["*"]` | replaced with env-driven `HSI_CORS_ALLOW_ORIGINS` | fixed |
| Healthcheck | minimal `/health` | now reports deploy/output/runtime mode info | improved |
| Output root | fixed `outputs/` under repo | moved to env-driven `HSI_OUTPUT_ROOT` with static mount prefix | fixed |
| Output URLs | assumed repo-relative `/outputs/...` | now derived from configured output root/prefix | fixed |
| Temporary uploads | `tempfile.TemporaryDirectory()` | acceptable for request-local upload staging | acceptable |
| Local server-side paths | accepted in public request path | disabled in public mode via env | fixed |
| Deploy config override | always allowed | public mode can reject override | fixed |
| Inference subprocess | defaulted to `conda run -n HyperSIGMA python` | now supports current interpreter by default, conda optional | fixed |
| HyperSIGMA import-time dependency | external repo resolved at import time | moved to lazy resolution during model load | fixed |
| `/outputs/...` on Render | tied to repo path assumption | works if `HSI_OUTPUT_ROOT` and static mount point stay aligned | acceptable with env |

## C. Model asset classification

| Asset | Role | Classification | Public handling |
| --- | --- | --- | --- |
| `model_best.pt` | real inference checkpoint, about 686 MB | `MUST_MOUNT` | place under `runtime/assets/models/E02_head_only_posw_v3_block224/model_best.pt` on Render |
| `temperature_scaling_val.json` | calibration freeze artifact, small and stable | `MUST_BUNDLE` | keep in repo and reference from the web deploy config |
| manifest | provenance / freeze reference, small and stable | `MUST_BUNDLE` | keep in repo under `annotations/manifests/...` |
| dataset | research training dataset, about 104 MB and not used at upload-time inference | `LOCAL_ONLY` | do not require it for public demo runtime; dataset path may be omitted in public provenance |

Additional non-listed but important assets:

- upstream `HyperSIGMA` repository: `MUST_MOUNT`
- upstream `spat-vit-b-checkpoint-1599.pth`: `MUST_MOUNT`
- upstream `spec-vit-b-checkpoint-1599.pth`: `MUST_MOUNT`

## D. Deployment blockers and risk items

| Risk | Why it matters | Current handling |
| --- | --- | --- |
| External HyperSIGMA repo dependency | current model loader still depends on upstream code | lazy import added, but runtime mount still required for real inference |
| Large checkpoint size | not appropriate to commit for public deploy path | moved to mounted `runtime/assets` strategy |
| CPU-only inference cost/latency | Render without GPU may be slow for HyperSIGMA | frontend production example defaults device to `cpu`; human validation still required |
| Ephemeral filesystem | Render loses local writes across deploys without a disk | `render.yaml` includes a persistent disk mount at `runtime/` |
| Public CORS domain | Netlify domain is deployment-specific | `HSI_CORS_ALLOW_ORIGINS` must be set manually on Render |
| Public local path exposure | `input_path` and path overrides are unsafe on public internet | public mode now rejects these inputs |
| Dataset provenance drift | training dataset should not become a hard public runtime dependency | dataset is treated as local-only in the public profile |

## Recommended public asset layout

```text
runtime/
  assets/
    models/
      E02_head_only_posw_v3_block224/
        model_best.pt
      baseline_v3_block224/
        model_best.pt
  upstream/
    HyperSIGMA/
      HyperspectralDetection/
        spat-vit-b-checkpoint-1599.pth
        spec-vit-b-checkpoint-1599.pth
        Target_Detection/
  outputs/
    preview_ui/
    web_ui/
```

## Recommendation

Adopt Netlify + Render for the first public demo only under these assumptions:

1. the backend is upload-first
2. runtime model assets are mounted outside Git
3. the first public validation is done with real Render health + preview + inference smoke checks
