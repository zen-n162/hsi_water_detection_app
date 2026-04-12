# HSI 水検出アプリケーション

このリポジトリは、ハイパースペクトル画像 (HSI) を入力として、水分子由来のスペクトル成分を検出し、その根拠となるアテンションマップを可視化するための実験用アプリケーションの実装です。主な対象センサは EO‑1/Hyperion や HISUI などの宇宙・航空搭載ハイパースペクトルセンサを想定しています。

## 特長

* **多様な入力フォーマット:** BigTIFF/GeoTIFF、ENVI (BSQ/BIL/BIP + `.hdr`)、HDF5 などの形式に対応できるよう設計されています。ヘッダファイルから波長情報を読み取り、波長–バンドの対応を保持します。
* **前処理パイプライン:** 不正バンドの除去、放射・反射率変換、タイル分割、幾何整合など、ハイパースペクトルデータ特有の前処理をスクリプト化して再現性を確保します。
* **HyperSIGMA 推論:** 事前学習済み HyperSIGMA モデルを用いて水検出を行い、各画素の水検出確率とともに空間方向・スペクトル方向のアテンションを取得します。
* **可視化:** 空間アテンションを RGB 合成画像に重ねて表示し、スペクトルアテンションを波長–重みのグラフとして出力します。
* **CLI/GUI 対応:** コマンドラインからのバッチ処理に加えて、将来的には簡易 GUI も搭載予定です。

## セットアップ

1. このリポジトリをクローンします。

   ```bash
   git clone <REPOSITORY_URL>
   cd hsi_water_detection_app
   ```

2. Conda 環境を用意する場合は `environment.yml` から環境を構築します。

   ```bash
   conda env create -f environment.yml
   conda activate hsi-water-detection
   ```

   もしくは、必要な依存パッケージを `requirements.txt` から直接インストールします：

   ```bash
   pip install -r requirements.txt
   ```

3. 必要に応じて HyperSIGMA の重みファイルを `models/` フォルダに配置します。重みファイルの取得方法については公式リポジトリや論文付録を参照してください。

## 使い方

CLI からの実行例を以下に示します。まずは `src/cli.py` を実行してください。

```bash
python -m src.cli \
    --input /path/to/your/hsi_file.tif \
    --output_dir /path/to/output \
    --header /path/to/your/hsi_file.hdr \
    --patch_size 64 --stride 32 \
    --model_checkpoint models/hyper_sigma.pth
```

主な引数の意味：

* `--input` – 入力となるハイパースペクトル画像ファイルのパス。
* `--output_dir` – 推論結果を保存するディレクトリ。
* `--header` – ENVI ヘッダファイル（波長やバンド情報を保持）へのパス。省略した場合はバンド番号に基づいてスペクトルアテンションをプロットします。
* `--patch_size` – モデルに与えるパッチのサイズ (デフォルト: 64)。
* `--stride` – パッチ分割時のストライド (デフォルト: 32)。
* `--model_checkpoint` – HyperSIGMA モデルの重みファイルへのパス。

出力として、以下のファイルが `--output_dir` に保存されます。

* `prob_map.tif` – 水検出確率マップ (GeoTIFF)。
* `spatial_attn.png` – 空間アテンションをカラーオーバーレイした画像。
* `spectral_attn.csv` – 各ターゲットピクセルに対する波長–アテンション重みの表。
* ほか、再構築した RGB 画像やメタデータ JSON など。

## 現在の進捗

このリポジトリは開発中です。現時点では、前処理および推論のためのスケルトンコードが `src/hsi_app.py` に含まれています。具体的なデータ読み込みや HyperSIGMA モデルの実装部分は未完成のため、利用の際には各自で関数を実装してください。

## 貢献

バグ報告やプルリクエストは歓迎します。実装の改善や新機能の追加など、ご協力いただける場合は issue を立ててご相談ください。
