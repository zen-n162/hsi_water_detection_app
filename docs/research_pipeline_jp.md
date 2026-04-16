# 研究パイプライン

この文書では、`confidence_ali.tif` をラベル source of truth とし、v3 block-split dataset に移行した後の、本リポジトリにおける湿潤・水関連 HyperSIGMA 研究パイプラインを説明します。

## 1. 目的

この研究パイプラインの目的は、**HyperSIGMA ベースの湿潤推論** を、再現可能・比較可能・デプロイ可能な形にすることです。

このパイプラインは次を支えます。

- truth raster から patch manifest を作る
- patch を export する
- baseline と HyperSIGMA を学習する
- validation に基づいて threshold を選ぶ
- temperature scaling で calibration する
- tuned threshold で test を再評価する
- baseline と比較する
- 実験をランキングし、次の実験候補を生成する
- freeze した最良 run を GUI / backend へ接続する

## 2. ラベルの source of truth

現在の source of truth の中心は次です。

- `confidence_ali.tif`
- `label_ali.tif`
- `gt_wetsoil_any.tif`
- `gt_wetsoil_highmid.tif`
- `gt_wetsoil_high.tif`
- `ignore_mask.tif`
- `eval_valid_mask.tif`

再構築後パイプラインでの重要な解釈は次です。

- `255`: ignore / invalid
- `0`: valid な dry-negative background
- `1, 2, 3`: wet-positive の confidence tier

つまり、現在のパイプラインは、旧 ROI manifest をラベル権威としては使っていません。

## 3. データセットの世代

### Legacy dataset
初期段階では、小規模な ROI ベース manifest から patch dataset を作っていました。初期試作には有効でしたが、label authority と grid alignment が十分に整理されていませんでした。

### v2 dataset
`confidence_ali.tif` からパイプラインを再構築し、以下を行いました。

- 新しい manifest 生成
- 新しい patch export
- 新しい train / val / test split 再計算
- 旧生成物と混ざらない出力先分離

### v3 dataset
現在の production-candidate 研究パイプラインでは、次を採用しています。

- block-based spatial split
- `block_size = 224`
- v2 より train / val / test 間の接触・重なり patch を削減
- calibration-aware な評価出力

v3 の入力は次です。

- Manifest: `annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv`
- Patch dataset: `datasets/processed/wetness_pretrain_v3`

## 4. Patch / manifest 生成ルール

再構築した manifest pipeline では、`confidence_ali.tif` から patch-level のラベルを導出します。

主要ルールは次です。

- Patch size: `64 x 64`
- 既定では非重複 patch 抽出
- `valid_ratio >= 0.90` の patch のみ採用
- `confidence_ali in {1,2,3}` を wet-positive とみなす
- `confidence_ali == 0` を dry-negative とみなす
- `255` は ignore
- `wet_ratio_valid >= 0.10` なら patch label は `wet`
- それ未満なら `dry`

manifest confidence は、旧 ROI ファイルからコピーするのではなく、patch 単位で再計算した要約値として保持します。

## 5. 学習パイプライン

## 5.1 Baseline
baseline は reference model として使う小型 CNN です。

典型的な流れ:
1. train split で学習
2. val split で評価
3. val から best threshold を選択
4. tuned threshold で test を評価
5. HyperSIGMA と比較

baseline は、attention 非依存の安定した reference path として重要です。

## 5.2 HyperSIGMA
HyperSIGMA 学習では、複数の fine-tuning 戦略を扱います。

- head-only tuning
- spectral-last partial unfreezing
- dual-last partial unfreezing
- progressive unfreezing

本研究で扱った代表的実験名は次です。

- `E01_head_only_bce`
- `E02_head_only_posw`
- `E05_spec_last_posw`
- `E07_dual_last_posw`
- `E09_progressive_posw`

現在 freeze 済みの production-candidate HyperSIGMA run は次です。

- `E02_head_only_posw_v3_block224`

## 6. 評価パイプライン

現在の評価スクリプトは次を扱えます。

- split 指定評価
- `--model_checkpoint` による fine-tuned checkpoint 読み込み
- sample-level 出力記録
- threshold ベース分類指標
- run 比較とランキング

主要指標は次です。

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1
- Confusion Matrix

v3 の calibration-aware 評価では、必要に応じて以下も扱います。

- Brier score
- ECE

## 7. Threshold tuning

threshold は validation prediction のみから決めます。

流れ:
1. val を評価
2. `samples[]` に per-sample score を保存
3. `select_threshold.py` を実行
4. best threshold JSON を保存
5. tuned threshold で test を再評価

これは、HyperSIGMA の score が極端に小さい側へ collapse しつつも、順位情報だけは残る場合があるため重要です。

## 8. Calibration

現在のパイプラインでは、**validation のみを使った temperature scaling** をサポートしています。

追加した理由:
- HyperSIGMA raw score が強く 0 側へ collapse していた
- ranking quality は残っていても probability quality が悪化していた
- raw collapsed score からの threshold 選択が不安定で解釈しにくかった

calibration の生成物には以下が含まれます。

- temperature JSON
- recalibrated evaluation
- 必要に応じた Brier score / ECE 集計

現在の production-candidate GUI path では calibrated HyperSIGMA v3 を使用しています。

## 9. Leakage audit と split 品質

データセット再構築後、leakage audit は重要工程になりました。

現在のパイプラインでは、明示的に以下を比較します。

- v2 split
- v3 block-split

patch-level dataset では、空間的に近い領域や重なりが train / val / test に跨ると、見かけ上の汎化性能が過大評価されるためです。

v3 block split は、この問題を v2 より実質的に抑えたため、現在の推奨方針です。

## 10. Ranking と best-run 選定

以下のユーティリティで実験管理を行います。

- `compare_models.py`
- `summarize_results.py`
- `make_next_experiments.py`

これらにより、
- baseline と HyperSIGMA を比較し
- 複数 HyperSIGMA run をランキングし
- 現時点の best run を選び
- 次の実験コマンドを自動生成できます

現在 freeze 済み構成では、
- best HyperSIGMA deployment candidate: `E02_head_only_posw_v3_block224`
- baseline は引き続き重要な競合 reference

## 11. GUI デプロイへの受け渡し

研究パイプラインの出力は、freeze 済み deploy config を介して GUI デプロイへ直接つながります。

この deploy config で固定されるもの:

- manifest
- dataset
- checkpoint
- temperature JSON
- threshold
- split policy
- run name

これにより、GUI は曖昧な手組み推論設定ではなく、再現可能な研究成果物に結びついた構成になります。

## 12. 現在の制約

- データセット規模はまだ大きくない
- HyperSIGMA の score collapse に対して calibration 対応が必要
- HyperSIGMA が deployability や PR-AUC で有利でも、運用点によっては baseline が強い場合がある
- spatial split 設計はまだ改善余地がある
- threshold policy も再設計の余地がある

## 13. 推奨される次の研究タスク

1. 単一 temperature scaling を超える calibration 改善
2. score collapse 診断を training log に直接追加
3. より強い spatially separated split の設計
4. baseline deploy path を GUI / backend に追加
5. 単純な best-F1 threshold を超える運用点選択の再検討
