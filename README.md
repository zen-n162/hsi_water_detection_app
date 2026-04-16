# hsi_water_detection_app

GUI-backed wetness and water-related hyperspectral inference workflow built around **HyperSIGMA** and a calibrated production-candidate deployment path.

This repository now contains both:

1. a **research pipeline** for building patch datasets, training/evaluating baseline and HyperSIGMA models, tuning thresholds, auditing split leakage, and calibrating scores, and  
2. a **GUI / backend inference application** that loads a frozen deploy configuration and runs ROI-based inference with provenance tracking.

## Current status

The current production-candidate GUI default is the **calibrated HyperSIGMA v3 block-split run**:

- Deploy config: `configs/deploy/hypersigma_v3_calibrated.json`
- Run name: `E02_head_only_posw_v3_block224`
- Model type: `ss`
- Patch size: `64`
- Stride: `32`
- Band count: `170`
- Label source: `confidence_ali.tif`
- Manifest: `annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv`
- Patch dataset: `datasets/processed/wetness_pretrain_v3`
- Checkpoint: `experiments/runs_v3/E02_head_only_posw_v3_block224/model_best.pt`
- Calibration: `experiments/runs_v3/E02_head_only_posw_v3_block224/temperature_scaling_val.json`
- Threshold: `0.327428693347738`

A calibrated HyperSIGMA GUI default was selected because it keeps the existing attention-capable inference path while preserving competitive ranking performance and improved calibration behavior relative to the uncalibrated run.

## What this project does

### Research side
- Rebuilds patch-level datasets from `confidence_ali.tif`
- Supports baseline CNN and HyperSIGMA fine-tuning experiments
- Evaluates on val/test with:
  - ROC-AUC
  - PR-AUC
  - Precision / Recall / F1
  - confusion matrix
  - optional Brier score / ECE in the v3 pipeline
- Tunes thresholds from validation predictions
- Fits validation-only temperature scaling
- Compares runs and summarizes experiment rankings
- Audits leakage risk under different split policies

### Application side
- Serves an ROI-based inference backend
- Supports deploy-config-driven default model loading
- Produces:
  - probability maps
  - probability overlays
  - spatial attention overlays
  - spectral attention plots
  - metadata / provenance JSON
- Exposes the deployed checkpoint, threshold, calibration file, manifest, dataset, split policy, and executed device through API responses and saved metadata

## Repository structure

```text
backend/
  app/
    api/
    services/
configs/
  deploy/
docs/
frontend/
research/
  datasets/
  evaluation/
  training/
src/
  hsi_water_detection_app/
annotations/
  manifests/
datasets/
  processed/
experiments/
  runs_v3/
  eval_v3/
outputs/
```

## Main documents

- [Research pipeline](docs/research_pipeline.md)
- [GUI inference contract](docs/gui_inference_contract.md)
- [Model card: HyperSIGMA v3 calibrated](docs/model_card_hypersigma_v3_calibrated.md)

## Quick start

### 1. Backend
```bash
cd /home/zennakamura/MasterResearch/hsi_water_detection_app
PYTHONPATH=src:. uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

### 2. Frontend
```bash
cd /home/zennakamura/MasterResearch/hsi_water_detection_app/frontend
npm run dev -- --host
```

### 3. Streamlit app (if used in your branch)
```bash
cd /home/zennakamura/MasterResearch/hsi_water_detection_app
python -m streamlit run ./src/hsi_water_detection_app/app.py
```

## Frozen production-candidate artifacts

The current GUI production-candidate freeze is documented through:

- `experiments/eval_v3/release_freeze.json`
- `experiments/eval_v3/release_verification.json`
- `experiments/eval_v3/runtime_versions.json`
- `experiments/eval_v3/gui_acceptance_result.json`
- `experiments/eval_v3/deploy_consistency_audit.json`
- `docs/final_hypersigma_gui_handover.md`

## Research highlights so far

### Data pipeline
- Source of truth moved from older ROI-centric labeling to `confidence_ali.tif`
- Patch manifest regenerated from raster truth
- v3 dataset uses **spatial block split** with `block_size=224`
- Leakage was reduced relative to earlier dataset versions

### Model evaluation
- Baseline remains an important reference path
- HyperSIGMA experiments E01 / E02 / E05 / E07 / E09 were organized and ranked
- `E02_head_only_posw_v3_block224` is the current best HyperSIGMA run for GUI deployment
- Calibration was added with validation-only temperature scaling

### Deployment / GUI integration
- GUI, backend, CLI, and metadata now resolve from a single deploy config
- Provenance is persisted in API responses and saved output metadata
- GPU execution was verified in the acceptance path
- The release candidate was frozen with a tagged Git commit

## Known limitations

- Dataset size is still small, so split policy strongly affects results
- HyperSIGMA score collapse required explicit calibration handling
- Baseline is currently a reference path and not the main GUI default inference route
- Frontend and backend should continue to be tested whenever deploy defaults are changed

## Recommended next work

1. Add a baseline GUI inference path alongside HyperSIGMA
2. Improve calibration beyond single-temperature scaling
3. Strengthen spatial split policy and leakage controls further
4. Add explicit score-collapse diagnostics to training/evaluation
5. Revisit threshold policy for more stable operating points

## Citation

If you use HyperSIGMA itself, cite the original HyperSIGMA paper and repository. The upstream project README is included in the repository history and was used as the foundation model reference during this integration work.
