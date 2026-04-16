# hsi_water_detection_app

**HyperSIGMA** を中心に構築した、GUI付きの湿潤・水関連ハイパースペクトル推論ワークフローです。校正済みの本番候補デプロイ経路まで含めて整理されています。

このリポジトリには現在、次の2系統が同居しています。

1. **研究パイプライン**  
   patch データセット構築、baseline / HyperSIGMA の学習・評価、threshold 調整、split leakage 監査、score calibration を行う流れ
2. **GUI / バックエンド推論アプリケーション**  
   freeze した deploy config を読み込み、ROI 単位の推論を provenance 付きで実行する流れ

## 現在の状態

現在の production-candidate GUI 既定モデルは、**calibrated HyperSIGMA v3 block-split run** です。

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

calibrated HyperSIGMA を GUI の既定にした理由は、既存の attention 対応推論経路を維持しつつ、uncalibrated run より calibration 挙動が改善され、ランキング性能も十分に競争力があったためです。

## このプロジェクトでできること

### 研究側
- `confidence_ali.tif` から patch-level データセットを再構築
- baseline CNN と HyperSIGMA の fine-tuning 実験
- val / test に対して以下を評価
  - ROC-AUC
  - PR-AUC
  - Precision / Recall / F1
  - confusion matrix
  - v3 では必要に応じて Brier score / ECE
- validation 予測から threshold を最適化
- validation のみを使った temperature scaling
- run 同士の比較とランキング集約
- split policy ごとの leakage リスク監査

### アプリケーション側
- ROI ベースの推論バックエンドを提供
- deploy config 駆動で既定モデルを解決
- 以下の出力を生成
  - probability map
  - probability overlay
  - spatial attention overlay
  - spectral attention plot
  - metadata / provenance JSON
- API 応答と保存 metadata の両方で、checkpoint、threshold、calibration file、manifest、dataset、split policy、実行 device を追跡可能

## リポジトリ構成

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

## 主なドキュメント

- [研究パイプライン](docs/research_pipeline.md)
- [GUI 推論契約](docs/gui_inference_contract.md)
- [モデルカード: HyperSIGMA v3 calibrated](docs/model_card_hypersigma_v3_calibrated.md)

## クイックスタート

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

### 3. Streamlit app（ブランチで使う場合）
```bash
cd /home/zennakamura/MasterResearch/hsi_water_detection_app
python -m streamlit run ./src/hsi_water_detection_app/app.py
```

## freeze された production-candidate artifacts

現在の GUI production-candidate freeze は、次のファイルで記録されています。

- `experiments/eval_v3/release_freeze.json`
- `experiments/eval_v3/release_verification.json`
- `experiments/eval_v3/runtime_versions.json`
- `experiments/eval_v3/gui_acceptance_result.json`
- `experiments/eval_v3/deploy_consistency_audit.json`
- `docs/final_hypersigma_gui_handover.md`

## ここまでの研究ハイライト

### データパイプライン
- ラベルの source of truth を旧 ROI 中心方式から `confidence_ali.tif` に移行
- patch manifest をラスタ truth から再生成
- v3 データセットでは `block_size=224` の **spatial block split** を採用
- 以前の dataset version と比べて leakage を低減

### モデル評価
- baseline は引き続き重要な reference path
- HyperSIGMA 実験 E01 / E02 / E05 / E07 / E09 を整理・比較
- GUI deployment 向けの最良 HyperSIGMA run は `E02_head_only_posw_v3_block224`
- validation-only temperature scaling による calibration を追加

### デプロイ / GUI 統合
- GUI / backend / CLI / metadata を単一 deploy config から解決する構成に整理
- provenance を API 応答と保存 metadata の両方に保持
- acceptance path で GPU 実行を確認
- release candidate を Git tag 付きで freeze

## 既知の制約

- データセット規模がまだ小さく、split policy の影響が大きい
- HyperSIGMA の score collapse に対して calibration 対応が必要
- baseline は現状 reference path であり、GUI の既定推論 route ではない
- deploy default を変更した場合、frontend / backend の両方を継続的にテストする必要がある

## 今後の推奨タスク

1. baseline GUI inference path を追加する
2. 単一 temperature scaling を超える calibration 改善を行う
3. spatial split policy と leakage 制御をさらに強化する
4. 学習・評価に score collapse 診断を追加する
5. より安定した運用点になるよう threshold policy を再検討する

## 引用

HyperSIGMA 自体を利用する場合は、元の HyperSIGMA 論文とリポジトリを引用してください。上流プロジェクトの README は本リポジトリ履歴にも含まれており、今回の統合作業における foundation model 参照元として利用しました。
