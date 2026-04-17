# hsi_water_detection_app

凍結済み HyperSIGMA deploy profile を中心にした、研究用パイプライン兼 Web デモアプリです。

このリポジトリには意図的に分離した 2 層があります。

1. データセット構築、学習、評価、校正、freeze artifact を扱う研究パイプライン
2. ROI preview / inference を provenance 付きで実行する Web 向け frontend/backend スタック

## 推奨公開構成

今回整備した公開構成の第一候補は次です。

- Frontend: Netlify
- Backend API: Render
- Runtime 資産: Render 側の `runtime/` mount

今回の整理で次を分離しました。

- 研究ローカル資産と研究ワークフロー
- upload-first の公開 Web 挙動
- 環境変数ベースの deployment 設定

参照:

- [Web deployment 監査](docs/web_deployment_audit.md)
- [Web deployment 概要](docs/web_deployment.md)
- [Netlify + Render 手順書](docs/netlify_render_deploy_guide.md)
- [Web release checklist](docs/web_release_checklist.md)

## 現在の freeze profile

ローカル研究/既定 profile:

- Config: `configs/deploy/hypersigma_v3_calibrated.json`
- Run: `E02_head_only_posw_v3_block224`
- Model type: `ss`
- Threshold: `0.327428693347738`

公開 Web profile:

- Config: `configs/deploy/hypersigma_v3_calibrated_web.json`
- Output root: `runtime/outputs`
- Mounted checkpoint root: `runtime/assets/models/...`

## Runtime mode

詳細は [Runtime modes](docs/runtime_modes.md) を参照してください。

要点:

- `local-research`: 研究用の学習・評価・freeze workflow
- `local-web-dev`: ローカル frontend/backend 開発。必要なら server-side path を使える
- `public-web`: 公開デモ用。upload 前提で path override を無効化

## ローカル開発

### Backend

```bash
cd /home/zennakamura/MasterResearch/hsi_water_detection_app
cp backend/.env.example backend/.env
PYTHONPATH=src:. uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

従来の conda subprocess 挙動を使いたい場合:

```bash
export HSI_INFERENCE_RUNTIME=conda
export HSI_INFERENCE_CONDA_ENV=HyperSIGMA
```

### Frontend

```bash
cd /home/zennakamura/MasterResearch/hsi_water_detection_app/frontend
cp .env.example .env.local
npm install
npm run dev -- --host
```

## 追加した公開向けファイル

- `frontend/.env.example`
- `frontend/.env.production.example`
- `frontend/netlify.toml`
- `backend/.env.example`
- `render.yaml`
- `configs/deploy/hypersigma_v3_calibrated_web.json`

## 重要な制約

- バックエンドは外部 HyperSIGMA repository を即時解決しなくても起動できるようになりましたが、実推論には runtime model assets の mount がまだ必要です。
- 公開モードでは server-side file path と deploy-config override を既定で無効化しています。
- 現行 HyperSIGMA 公開デモ経路は、大きな fine-tuned checkpoint と上流 weights に依存するため、CPU/GPU 環境差分の影響を受けます。

## 主なドキュメント

- [GUI inference contract](docs/gui_inference_contract.md)
- [Runtime modes](docs/runtime_modes.md)
- [Research pipeline](docs/research_pipeline.md)
- [Model card: HyperSIGMA v3 calibrated](docs/model_card_hypersigma_v3_calibrated.md)
