# モデルカード: hypersigma_v3_calibrated

## モデル名

- `hypersigma_v3_calibrated`

## モデルの役割

このモデルは、GUI / backend の production candidate inference path として使うことを想定した、**校正済み HyperSIGMA v3** です。

主な目的:
- Hyperion 画像上で ROI ベース湿潤推論を行う
- probability map と attention 系 artifact を出す
- provenance を deploy config 起点で一貫して記録する

## 想定用途

- GUI からの ROI 推論
- backend API からの推論
- calibrated score と tuned threshold を用いた運用
- 研究成果物と接続された可視化付き inference path

## 非想定用途

- 他 sensor への無条件適用
- calibration なし raw score の運用
- baseline path の代替定義
- 将来の最終完成版モデルの宣言

## 入力仕様

- Sensor: `hyperion`
- Patch size: `64 x 64`
- Stride: `32`
- Band count: `170`
- Model type: `ss`

## 学習データの由来

ラベル source of truth:
- `confidence_ali.tif`

データセット系:
- Manifest: `annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv`
- Patch dataset: `datasets/processed/wetness_pretrain_v3`

## Split policy

- `spatial_block(block_size=224)`

この split policy は、以前の dataset version と比べて patch-level leakage を減らすために採用されました。

## 学習 run の識別子

freeze された deployment run:

- Run name: `E02_head_only_posw_v3_block224`
- Checkpoint: `experiments/runs_v3/E02_head_only_posw_v3_block224/model_best.pt`

## Calibration

calibration は有効です。

- Method: validation-only temperature scaling
- File: `experiments/runs_v3/E02_head_only_posw_v3_block224/temperature_scaling_val.json`

これを導入した理由は、raw HyperSIGMA score が非常に小さい値側へ強く collapse し、uncalibrated probability が deployment に不安定だったためです。

## 運用 threshold

- Threshold source: validation-selected calibrated threshold
- Threshold value: `0.327428693347738`

この threshold は、calibrated score path と組み合わせて使う前提です。raw uncalibrated score path にそのまま適用するものではありません。

## 主要評価指標

v3 calibrated test 評価での主指標:

- ROC-AUC: `0.7333`
- PR-AUC: `0.5917`
- F1: `0.4000`

解釈:
- ranking quality は実用候補レベルにある
- calibration は uncalibrated run より改善している
- attention 対応出力と deploy path を保ったまま GUI default に載せられる
- ただし operating-point の安定性にはまだ課題がある

## 比較上の位置づけ

baseline reference path は引き続き重要です。

現時点の整理:
- baseline は一部観点では依然として強い
- それでも HyperSIGMA は、既存の attention-based inference / deploy 構造に自然に載せられる点を重視して、GUI production candidate として採用している

## 既知の failure modes

- calibration なしでは raw score が強く 0 側へ collapse する
- データセット規模が小さく、split 設計への感度が高い
- threshold choice により分類性能の見え方が大きく変わる
- temperature scaling 後も probability calibration は完全ではない
- ranking quality が残っていても、default threshold 挙動が悪い場合がある

## Deployment provenance

GUI / backend deployment path は常に次から解決されるべきです。

- `configs/deploy/hypersigma_v3_calibrated.json`

保存済み GUI / backend outputs では、次を通じて provenance が追跡できる必要があります。

- API response fields
- `metadata.json`
- deploy consistency audit outputs
- release freeze / release verification records

## 推奨モニタリング項目

今後の run では少なくとも以下を監視することが望ましいです。

- ROC-AUC
- PR-AUC
- tuned threshold での F1
- score distribution の形
- Brier score
- ECE
- score collapse の兆候
- split leakage 指標

## 推奨される次の改善

1. 単一 temperature scaling を超える calibration 改善
2. 学習・評価に明示的な score collapse 診断を追加
3. spatial split policy をさらに強化
4. baseline GUI inference path を追加し、並行比較可能にする
5. より安定した実運用 threshold policy を再設計する
