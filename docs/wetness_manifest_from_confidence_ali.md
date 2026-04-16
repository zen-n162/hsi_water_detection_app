# Wetness Manifest From `confidence_ali.tif`

このデータ生成系では、旧 `wetness_manifest.csv` の ROI 座標を正解として使いません。
唯一の ground truth source は `confidence_ali.tif` です。

## Source of Truth

- ラベル raster: `Hyperion_WaterLabel_20111222/data/processed/labels/confidence_ali.tif`
- patch 抽出元 cube: `Hyperion_WaterLabel_20111222/data/processed/hyperion/hyperion_stack_crop_f16.tif`
- 両者は shape / CRS / transform が一致する同一グリッドです。

## Fixed Rules

- patch size: `64x64`
- stride: `64` の非重複 patch
- valid pixel: `confidence_ali != 255`
- ignore patch: `valid_ratio < 0.90`
- wet-positive pixel: `confidence_ali in {1,2,3}`
- dry-negative pixel: `confidence_ali == 0`
- patch label:
  - `wet_ratio_valid >= 0.10` なら `wet`
  - それ未満なら `dry`

## Manifest Confidence Mapping

- wet patch:
  - `mean_tier = mean(confidence_ali values over wet pixels)`
  - `manifest_confidence = valid_ratio * (mean_tier / 3.0)`
- dry patch:
  - `manifest_confidence = valid_ratio * (1 - wet_ratio_valid)`

補助列として `confidence_tier` も保持します。

- wet patch: patch 内 wet pixel の dominant tier (`1/2/3`)
- dry patch: `0`

## Split Policy

- 旧 manifest は参照しません。
- 新 manifest 上で `label` に対する stratified split を再計算します。
- 既定比率は `train=0.70`, `val=0.15`, `test=0.15`
- 既定 seed は `42`

## Outputs

- manifest:
  - `annotations/manifests/wetness_manifest_from_confidence_ali.csv`
- summary:
  - `annotations/manifests/wetness_manifest_from_confidence_ali.summary.json`
- exported patches:
  - `datasets/processed/wetness_pretrain_v2`

## Re-run

```bash
cd ~/MasterResearch/hsi_water_detection_app
PYTHONPATH=src:. python research/datasets/build_wetness_manifest_from_confidence_ali.py
PYTHONPATH=src:. python research/datasets/export_roi_patches.py \
  --manifest annotations/manifests/wetness_manifest_from_confidence_ali.csv \
  --output_dir datasets/processed/wetness_pretrain_v2
```
