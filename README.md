# Audio Algorithm — 音声AI技術選定・評価ポートフォリオ

美律のSoftware & Algorithm Manager職で要求される、技術選定、評価設計、製品化判断を説明するための研究用プロトタイプです。企業の公式成果物ではありません。

**実装済み:** 著者公式GTCRNへのオフライン推論アダプタ、公式TorchAudioによるマスク入力MVDR、データ分割監査、SI-SDR改善量、条件別集計、CPUベンチマーク、単体テスト。

**未実施:** 独自学習、論文ベンチマークの再現、実話者での有効性評価、AEC・ANC・風雑音専用モデル、補聴処理、組込みストリーミング、実チップ電力測定、EVT/DVT/PVT、認証。デモの高い数値を製品性能として提示しないでください。

- [技術比較・3つの原因検証計画・関数別Big-O](docs/technical_report_ja.md)
- [面接説明・90日計画・量産ゲート](docs/interview_ja.md)
- [一次資料一覧と適用範囲](docs/sources.md)
- [検証結果](evidence/validation.md)

## インストールと実行

Python 3.12の専用環境を推奨。検証したCPU版を固定しています。対象OSで同じwheelが得られない場合は依存を勝手に置換せず、公式配布元で互換性を確認してください。

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.14.1+cpu torchaudio==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python audio_lab.py demo
.venv\Scripts\python audio_lab.py audit examples/manifest.json
.venv\Scripts\python audio_lab.py summarize examples/scores.json --conditions indoor wind
```

`demo`は合成正弦波と理想マスクによる数式・配線の確認です。音声品質の比較試験ではありません。MVDR単体APIの入力は複素STFT `[C,F,T]`、音声・雑音マスクはそれぞれ `[F,T]`。実録音用マスク推定器は含みません。

## 公式GTCRNで16 kHzモノラル音声を処理

第三者コードと重みは本リポジトリに同梱しません。著者の固定コミットを取得し、利用する音声の権利を確認してください。`third_party`はGit管理対象外です。

```powershell
git clone https://github.com/Xiaobin-Rong/gtcrn.git third_party/gtcrn
git -C third_party/gtcrn checkout --detach 502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73
.venv\Scripts\python audio_lab.py enhance input.wav enhanced.wav --gtcrn-source third_party/gtcrn --checkpoint third_party/gtcrn/checkpoints/model_trained_on_dns3.tar
.venv\Scripts\python audio_lab.py score clean.wav input.wav enhanced.wav
```

PCM16 WAV、最大60秒。出力先は新規ファイル限定。クリッピング時は黙って正規化せずエラーにします。`score`には遅延補正済みの同長・同サンプルレートの音声を渡してください。参照音声は評価専用で、GTCRN推論へ渡しません。無音の参照・推定音声はSI-SDR未定義として例外にし、バッチ評価では失敗として記録してください。

`load_gtcrn`は指定フォルダのPythonを実行します。信頼する公式コミットだけを指定してください。モデルは一度だけロードし、複数発話には同じモデルを再利用できます。現アダプタは全発話単位であり、長時間録音を単純分割してストリーミング同等だとは主張しません。

## 評価データ

`examples`は架空の説明用メタデータです。実データ評価では、`speaker_id/session_id/source_id`をデータセット全体で一意にし、切り出し・増強前の元音源を追跡してください。`sha256`は実際のファイルから計算します。監査は申告されたIDとハッシュの整合性検査で、音声の近似重複や基盤モデルの学習集合混入は検出しません。

比較は同一テスト集合・同一参照・同一前処理で行い、スコアファイルに各録音の`condition`と`delta_db`を記録します。失敗は`null`で残します。条件は事前に宣言し、欠落条件・失敗がある場合の`complete=false`を合格扱いしないでください。集計はサンプル平均であり、話者ごとの信頼区間は別途必要です。

## 公開範囲

新規作成コード・資料は[MIT](LICENSE)。引用先のコード・モデル・データのライセンスは別です。第三者音声、モデル重み、個人情報、企業機密は同梱しません。製品適合を保証する資料ではありません。
