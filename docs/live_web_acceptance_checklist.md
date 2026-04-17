# Live Web Acceptance Checklist

This document defines the release acceptance procedure once Netlify and Render URLs exist.

## What is automated

The Python script:

- checks frontend root reachability
- checks backend `/health`
- checks backend `/inference/deploy-config`
- uploads a live sample to `/preview/grayscale`
- uploads a live sample to `/inference/run`
- fetches generated result assets
- fetches `metadata.json`
- validates provenance keys such as:
  - `resolved_model_checkpoint`
  - `resolved_temperature_json`
  - `resolved_threshold`
  - `executed_device`

Script path:

- [research/evaluation/live_web_acceptance_checklist.py](/home/zennakamura/MasterResearch/hsi_water_detection_app/research/evaluation/live_web_acceptance_checklist.py)

## What remains manual

Because the repository does not bundle a browser automation stack, these UI checks remain manual in a real browser:

- top page layout is visually correct
- deploy info is visibly shown in the `Resolved inference settings` panel
- ROI selection box is visible and draggable
- Pseudo Color / Overlay / Spatial / Spectral cards render in the browser
- model provenance is visible in the sidebar

## Required inputs

- `frontend_url`
- `backend_url`
- `sample_hsi`
- optional `sample_wavelength`

Recommended sample:

- a small Hyperion crop or another public-safe HSI sample that is valid for the deployed sensor mode
- pass it explicitly with `--sample-hsi`; the script intentionally has no machine-specific default path

## Example command

```bash
cd /home/zennakamura/MasterResearch/hsi_water_detection_app
python research/evaluation/live_web_acceptance_checklist.py \
  --frontend-url https://<your-netlify-site>.netlify.app \
  --backend-url https://<your-render-service>.onrender.com \
  --sample-hsi /absolute/path/to/hyperion_stack_crop_f16.tif \
  --sensor hyperion \
  --device cpu
```

Optional wavelength sidecar:

```bash
python research/evaluation/live_web_acceptance_checklist.py \
  --frontend-url https://<your-netlify-site>.netlify.app \
  --backend-url https://<your-render-service>.onrender.com \
  --sample-hsi /absolute/path/to/hyperion_stack_crop_f16.tif \
  --sample-wavelength /absolute/path/to/wavelengths.json
```

## Execution order

1. open the frontend root manually
2. confirm the page title is `HSI Water Detection UI`
3. run the acceptance script
4. inspect the output JSON from the script
5. perform the manual browser-only checks below

## Manual browser-only checks

### Frontend

- [ ] top page loads without console-visible fatal error
- [ ] `Resolved inference settings` panel is visible
- [ ] deploy config info is visible in that panel
- [ ] upload controls are visible
- [ ] after preview, grayscale image is shown
- [ ] ROI selection area is visible and draggable
- [ ] after inference, results page shows:
  - Pseudo Color
  - H2O Detection Overlay
  - Spatial Attention
  - Spectral Attention
- [ ] provenance fields are visible in the side panel

### Backend

- [ ] `/health` succeeded
- [ ] `/preview/grayscale` succeeded
- [ ] `/inference/run` succeeded
- [ ] `metadata.json` was reachable
- [ ] `resolved_model_checkpoint` was present
- [ ] `executed_device` was present

## Pass criteria

The live release is accepted only if:

1. the script exits with code `0`
2. all generated asset URLs are reachable
3. `metadata.json` includes the required provenance
4. manual browser checks are all marked complete
