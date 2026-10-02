# 実行検証記録

検証日: 2026-10-02。Windows / Python 3.12.14 / PyTorch 2.14.1+cpu / TorchAudio 2.11.0+cpu。依存一覧は[environment.txt](environment.txt)。

|検証|結果|何を証明しないか|
|---|---|---|
|unittest 10件|全件成功。SI-SDRスケール不変性、無効入力拒否、各分割軸の漏洩検出、少数条件の埋没、欠落/失敗の可視化、PCM入出力、マイク置換整合、マスクなしfallback、STFT復元、benchmark呼出回数|実音声・対象製品での品質|
|GTCRN公式DNS3重み|ロード・推論成功。16,017サンプル入力と同長の有限値出力。eval・勾配無効。公式レシピを現行複素APIで表した参照計算との差0|論文のDNS/VCTK評価の再現、未知話者への汎化|
|GTCRNパラメータ計測|総数48,245、学習可能23,669|MACs、RAM、電力|
|GTCRN CLI WAV入出力|自作合成入力で成功。入力・出力・コード・重みのSHA-256を出力|音声内容保持、聴取品質|
|MVDR合成理想マスク|数値が有限、処理パイプラインが動作|実マスク推定、リアルタイム、製品性能|

[gtcrn_smoke.json](gtcrn_smoke.json)のCPU計測は1スレッド、warmup 3回、20回反復、約1秒の合成波形に対し中央値約53.24 ms、p95約65.84 ms、RTF約0.0532。GTCRNのSTFT・推論・逆STFTを含み、重みロードとファイルI/Oを含みません。初回起動、実時間入力、ADC/DAC、Bluetooth、チップ電力は測定していません。p95は20標本の順位統計であり、高精度なtail推定ではありません。

[synthetic_oracle_demo.json](synthetic_oracle_demo.json)は、理想マスク・反位相雑音という人工的条件です。SI-SDRは入力約−0.04 dB、出力約18.94 dBですが、これを音声品質改善として提示してはいけません。benchmarkは事前計算済みSTFT/マスクからMVDRと逆STFTまでで、GTCRNと計測範囲が異なるため速度順位には使えません。

## 再実行

READMEの環境準備とGTCRNの固定コミット取得後、リポジトリルートから実行します。

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python tools/validate_synthetic.py --gtcrn-source third_party/gtcrn --output-dir local_results
```

`local_results`には自作合成入力とJSONが生成されます。`save`はJSONサイズJに対し時間・追加空間O(J)。スクリプト全体の計算量は呼び出すデモ・GTCRN・benchmarkの和です。乱数seedは固定していますが、異なるハードウェア/版でのbit一致や時間一致は保証しません。

メモリの長時間soak試験、GPU、実音声評価、PESQ/POLQA/NISQA、学習、量子化、ストリーミング、実チップ、AEC/ANC、製造検証は未実施です。
