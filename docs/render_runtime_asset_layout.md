# Render Runtime Asset Layout

This document fixes the runtime asset layout for the public Render deployment.

## Mount path

Render persistent disk mount path:

- `/opt/render/project/src/runtime`

Application-level runtime root:

- `HSI_RUNTIME_ROOT=runtime`

This means repository-relative runtime paths resolve to:

- `runtime/assets/...`
- `runtime/outputs/...`

## Canonical directory tree

```text
runtime/
  assets/
    models/
      E02_head_only_posw_v3_block224/
        model_best.pt
        temperature_scaling_val.json
      baseline_v3_block224/
        model_best.pt
    manifests/
      wetness_manifest_from_confidence_ali_blocksplit.csv
    datasets/
      wetness_pretrain_v3_minimal/
        README.txt
        provenance_manifest.txt
    upstream/
      HyperSIGMA/
        HyperspectralDetection/
          spat-vit-b-checkpoint-1599.pth
          spec-vit-b-checkpoint-1599.pth
          Target_Detection/
          ...
  outputs/
    preview_ui/
    web_ui/
```

## Files that must be present

Required for the public default path:

- `runtime/assets/models/E02_head_only_posw_v3_block224/model_best.pt`
- `runtime/assets/models/E02_head_only_posw_v3_block224/temperature_scaling_val.json`
- `runtime/assets/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv`
- `runtime/assets/upstream/HyperSIGMA/HyperspectralDetection/spat-vit-b-checkpoint-1599.pth`
- `runtime/assets/upstream/HyperSIGMA/HyperspectralDetection/spec-vit-b-checkpoint-1599.pth`
- `runtime/assets/upstream/HyperSIGMA/HyperspectralDetection/Target_Detection/...`

Optional but recommended:

- `runtime/assets/models/baseline_v3_block224/model_best.pt`
- `runtime/assets/datasets/wetness_pretrain_v3_minimal/`

## Deploy-config references

The public production candidate resolves these paths from [configs/deploy/hypersigma_v3_calibrated_web.json](/home/zennakamura/MasterResearch/hsi_water_detection_app/configs/deploy/hypersigma_v3_calibrated_web.json):

- `model_checkpoint` -> `runtime/assets/models/E02_head_only_posw_v3_block224/model_best.pt`
- `temperature_json` -> `runtime/assets/models/E02_head_only_posw_v3_block224/temperature_scaling_val.json`
- `manifest_path` -> `runtime/assets/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv`
- `dataset_path` -> `runtime/assets/datasets/wetness_pretrain_v3_minimal`
- `output_root` -> `runtime/outputs`

The backend also uses:

- `HSI_RUNTIME_ROOT=runtime`
- `HYPERSIGMA_ROOT=runtime/assets/upstream/HyperSIGMA`

## Initial upload procedure

### 1. Create the Render web service and attach the disk

Do this first so the mount path exists.

### 2. First deploy the backend

Let the service boot once with the disk mounted.

### 3. Open the Render shell and create the directory tree

```bash
mkdir -p /opt/render/project/src/runtime/assets/models/E02_head_only_posw_v3_block224
mkdir -p /opt/render/project/src/runtime/assets/models/baseline_v3_block224
mkdir -p /opt/render/project/src/runtime/assets/manifests
mkdir -p /opt/render/project/src/runtime/assets/datasets/wetness_pretrain_v3_minimal
mkdir -p /opt/render/project/src/runtime/assets/upstream
mkdir -p /opt/render/project/src/runtime/outputs/preview_ui
mkdir -p /opt/render/project/src/runtime/outputs/web_ui
```

### 4. Transfer the files

Recommended methods:

1. SCP/SFTP using the SSH connection details shown in the Render dashboard
2. Magic Wormhole from the Render shell if SSH is not set up

Local source files:

```text
experiments/runs_v3/E02_head_only_posw_v3_block224/model_best.pt
experiments/runs_v3/E02_head_only_posw_v3_block224/temperature_scaling_val.json
annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv
```

Local destination paths:

```text
/opt/render/project/src/runtime/assets/models/E02_head_only_posw_v3_block224/model_best.pt
/opt/render/project/src/runtime/assets/models/E02_head_only_posw_v3_block224/temperature_scaling_val.json
/opt/render/project/src/runtime/assets/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv
```

For the upstream repository, copy or clone the required `HyperSIGMA` tree under:

```text
/opt/render/project/src/runtime/assets/upstream/HyperSIGMA
```

### 5. Create the minimal dataset placeholder

The public inference path does not need the full training dataset, but the public deploy profile carries a minimal provenance pointer.

Create:

```text
/opt/render/project/src/runtime/assets/datasets/wetness_pretrain_v3_minimal/README.txt
/opt/render/project/src/runtime/assets/datasets/wetness_pretrain_v3_minimal/provenance_manifest.txt
```

Suggested contents:

- source dataset name
- freeze date
- git commit/tag
- note that the full training dataset stays local-only

## Update procedure

### Updating only the model checkpoint

1. upload the new checkpoint into the same model directory with a versioned filename first
2. verify checksum and file size
3. replace `model_best.pt` only after validation
4. restart the backend service
5. rerun live acceptance

### Updating calibration JSON

1. upload the new JSON alongside the existing one
2. if it becomes the official production candidate, update the public deploy config or set `HSI_TEMPERATURE_JSON`
3. rerun live acceptance

### Updating the upstream HyperSIGMA tree

1. upload into a new sibling directory, for example `HyperSIGMA_2026-04-17`
2. validate imports and weights
3. switch `HYPERSIGMA_ROOT`
4. rerun live acceptance

## Files that must not be committed to the public repo

- `runtime/assets/models/E02_head_only_posw_v3_block224/model_best.pt`
- `runtime/assets/upstream/HyperSIGMA/**`
- upstream pretrained encoder weights if not already public and intentionally mirrored
- any copied runtime disk artifacts
- generated `runtime/outputs/**`

These remain outside Git and live only on the Render disk or another controlled artifact store.
