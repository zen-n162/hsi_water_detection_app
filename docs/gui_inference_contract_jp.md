# GUI 推論契約

この文書では、calibrated HyperSIGMA v3 GUI 経路の deploy-ready inference contract を定義し、frontend、backend、CLI、freeze 済み研究成果物がどのように接続されるかを明確化します。

## 1. Deploy default

現在の GUI production-candidate default は、次から解決されます。

- Deploy config: `configs/deploy/hypersigma_v3_calibrated.json`
- Run name: `E02_head_only_posw_v3_block224`
- Model type: `ss`
- Manifest: `annotations/manifests/wetness_manifest_from_confidence_ali_blocksplit.csv`
- Patch dataset: `datasets/processed/wetness_pretrain_v3`
- Checkpoint: `experiments/runs_v3/E02_head_only_posw_v3_block224/model_best.pt`
- Temperature scaling: `experiments/runs_v3/E02_head_only_posw_v3_block224/temperature_scaling_val.json`
- Threshold: `0.327428693347738`
- Split policy: `spatial_block(block_size=224)`

frontend の default は、ファイルごとの固定値ではなく、この deploy config から導出される必要があります。

## 2. システムフロー

デプロイ済み推論経路は次です。

1. **Frontend** が deploy default を要求する
2. **Backend** が deploy config を具体的な artifact path / 値へ解決する
3. **Frontend** が ROI 推論リクエストを送る
4. **Backend** が推論経路を起動し、解決済み provenance を記録する
5. **CLI / inference code** が選択モデルで ROI 推論を実行する
6. **出力 artifact** が `outputs/web_ui/<timestamp>/` に保存される
7. **API response + metadata.json** の両方で、resolved checkpoint / threshold / calibration / run identity を保持する

この契約は、保存済み出力ディレクトリだけで「どのモデルと設定を使ったか」を再構築できることを意図しています。

## 3. GET `/inference/deploy-config`

目的:
- frontend に、現在の deploy-ready default model configuration を渡す

任意 query param:
- `deploy_config_path`

省略時は backend は次を使います。
- `configs/deploy/hypersigma_v3_calibrated.json`

応答には最低でも以下を含めるべきです。
- `deploy_config_path`
- `deploy_config_relative`
- `deploy_config`
- `resolved`
- `reference_models`

`resolved` は deploy config を backend が具体化した結果です。

## 4. POST `/inference/run`

必須 request fields:

- `hsi_file`
- `sensor`
- `device`
- `row_start`
- `row_stop`
- `col_start`
- `col_stop`

任意 override fields:

- `deploy_config_path`
- `wavelength_file`
- `model_type`
- `model_checkpoint`
- `temperature`
- `temperature_json`
- `decision_threshold`
- `manifest_path`
- `patch_dataset_path`
- `split_policy`
- `spat_checkpoint`
- `spec_checkpoint`
- `patch_size`
- `stride`
- `xmin`
- `ymin`
- `xmax`
- `ymax`

解決ルール:

1. backend はまず deploy config を読み込む
2. request override が非空なら deploy default を上書きする
3. 解決後に `model_checkpoint` が必ず存在していなければならない
4. `temperature_json` が解決された場合は calibration を適用する
5. `decision_threshold` が無ければ deploy-config threshold を使う
6. backend は requested device と executed device の両方を記録する

## 5. API response における安定 provenance fields

backend response には、次の安定キーを保持するべきです。

- `deploy_config_path`
- `resolved_model_checkpoint`
- `resolved_temperature_json`
- `resolved_temperature`
- `resolved_threshold`
- `resolved_manifest`
- `resolved_dataset`
- `resolved_split_policy`
- `resolved_run_name`
- `resolved_band_count`
- `resolved_model_type`
- `resolved_patch_size`
- `resolved_stride`
- `requested_device`
- `executed_device`
- `model_provenance`
- `files`
- `urls`

`model_provenance` には次を含めるべきです。

- `deploy_config_path`
- `deploy_name`
- `run_name`
- `manifest_path`
- `patch_dataset_path`
- `model_checkpoint_path`
- `calibration_file_path`
- `temperature`
- `threshold`
- `split_policy`
- `band_count`
- `reference_models`

## 6. Output directory contract

各 GUI / backend inference run は次に保存されます。

- `outputs/web_ui/<timestamp>/`

期待される artifacts:

- `probability_map.png`
- `probability_overlay.png`
- `spatial_attention_overlay.png`
- `spectral_attention.png`
- `metadata.json`
- `run_config.json`
- `api_result.json`

`metadata.json` は、保存された provenance の正規 artifact です。

最低でも以下を含める必要があります。
- resolved checkpoint
- resolved threshold
- resolved temperature / calibration file
- run name
- manifest path
- dataset path
- split policy
- requested device
- executed device

## 7. Frontend 側の期待事項

frontend では、次が見えることが望まれます。

- run name
- checkpoint path または短縮 checkpoint id
- threshold
- calibration 状態
- deploy config path
- 実行 device

これは、「どの研究成果物で今 GUI が動いているのか」を user / developer が即座に把握できるようにするためです。

## 8. Calibration contract

calibrated HyperSIGMA v3 default では、次を前提にします。

- score は raw logits から直接 threshold しない
- validation-only で fit した temperature を使う
- GUI default threshold は calibrated validation best threshold を使う
- threshold は uncalibrated path と混用しない

## 9. HyperSIGMA と baseline の関係

現行 production path は HyperSIGMA です。

- HyperSIGMA: production-candidate GUI default
- baseline: reference-only path

baseline GUI path は将来追加候補ですが、現時点では本 contract の主対象ではありません。

## 10. Acceptance 条件

「GUI 接続完了」と見なすための最低条件:

- deploy config default load が成功する
- frontend が run name / checkpoint / threshold / calibration を表示する
- ROI inference が成功する
- `probability_overlay.png` が保存される
- `metadata.json` が保存される
- metadata に resolved checkpoint / threshold / calibration / run name が含まれる
- executed device が記録される
- consistency audit が overall_ok になる
- acceptance result が overall_ok になる

## 11. 実務上の注意

- deploy config を変えたら frontend / backend / CLI / smoke test を再実行する
- threshold と temperature JSON は checkpoint とセットで扱う
- manifest / dataset / split policy を勝手に差し替えない
- release freeze 前には provenance を JSON で固定する
