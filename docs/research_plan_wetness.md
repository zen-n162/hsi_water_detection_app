# Wetness Pretraining Research Plan

## Goal
Earth-side wetness proxy labelsからHyperSIGMAベースのwetness detectorを学習し、
そのcheckpointを既存のHSI Water Detection UIへ流し込んで比較・可視化する。

## Phase 1
- wetness_manifest.csv を作る
- ROI patch を export する
- baseline を学習する
- HyperSIGMA 版 wetness pretrain を回す
- AUC / PR-AUC / attention diagnostics を評価する

## Labels
- wet
- dry
- uncertain

## Sensors
- hyperion
- hisui
- m3
- iirs
