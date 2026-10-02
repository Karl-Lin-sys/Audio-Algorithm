# 一次資料・適格性台帳

調査日: 2026-10-02。技術選定の論文期間は2023-10-02〜2026-10-02、原因分析の期間は2021-10-02〜2026-10-02。検索結果に表示された一般ブログ・フォーラムは根拠に採用していません。網羅的な2026年最新SOTA調査ではありません。

「CVPR、ICCV、ECCV、NeurIPS等」の「等」に、音声分野の主要査読会議ICASSP/Interspeechを含める解釈です。CV系会議だけを厳密指定する場合、GTCRNは対象外になります。論文の適格性とGitHubの人気・再現可能性は別々に評価します。

|ID|資料|用途・制約|
|---|---|---|
|P1|Rong et al., [GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources, ICASSP 2024](https://ieeexplore.ieee.org/document/10448310)|査読会議・期間内。著者公式READMEからも書誌を照合。公開実装の値と論文値を区別|
|P2|Babaev et al., [FINALLY, NeurIPS 2024 Main Track](https://proceedings.neurips.cc/paper_files/paper/2024/hash/01b3dea1871f7cea1e0e6be1f2f085bc-Abstract-Conference.html)|査読済み・期間内。高音質復元の比較対象。今回公式実装を確認できず導入対象から除外|
|P3|Yang et al., [RealMAN, NeurIPS 2024 Datasets and Benchmarks Track](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bf8f6f5b017dc60d0c4e28a7a9a4ee7b-Abstract-Datasets_and_Benchmarks_Track.html)|査読済み・期間内。実収録マイクアレイ評価の候補。データセットであり強調モデルではない|
|P4|Zhang et al., [URGENT, Interspeech 2024](https://www.isca-archive.org/interspeech_2024/zhang24h_interspeech.pdf)|査読済み・期間内。タスク・入力・指標の多様性の根拠。NeurIPS競技採択を本会議論文と混同しない|
|P5|Kapoor & Narayanan, [Leakage and the reproducibility crisis in machine-learning-based science, Patterns 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/)|査読誌・5年以内。3年条件の技術候補としては使用しない。分割、サンプリング、評価の原因分析に限定|
|C1|[GTCRN公式実装](https://github.com/Xiaobin-Rong/gtcrn/tree/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73)|MIT。モデル構成・infer.py・READMEを確認。コミット固定で動作検証|
|C2|[ESPnet公式実装](https://github.com/espnet/espnet/tree/f801d34fb0de1d14a6c763c587bc7ea7bb990ccb)|Apache-2.0。音声実験recipe基盤として比較。本体を今回インストール・検証してはいない|
|C3|[RealMAN公式公開元](https://github.com/Audio-WestlakeU/RealMAN/tree/9dc59f03a98149bc7fe5524d1363af83a5b399f7)|READMEはデータCC-BY-4.0を明記。APIのrepository licenseはnullであり、コードの包括許諾を意味しない|
|D1|[TorchAudio公式MVDRチュートリアル](https://docs.pytorch.org/audio/stable/tutorials/mvdr_tutorial.html)|PSD→SoudenMVDR→逆STFTというAPI構成の根拠。理想マスクを実運用性能と混同しない|
|D2|[SoudenMVDR API](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.SoudenMVDR.html)|diagonal loadingと入出力仕様。数学基準法であり最近の新規論文としては扱わない|
|D3|[PyTorch inference_mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)|推論時グラフ抑制。eval()は別に必要|
|D4|[PyTorch STFT](https://docs.pytorch.org/docs/stable/generated/torch.stft.html)、[ISTFT](https://docs.pytorch.org/docs/stable/generated/torch.istft.html)|複素テンソル、明示window、length指定。実環境でround-tripテスト|
|D5|[TensorFlow tf.function](https://www.tensorflow.org/guide/function)|再トレース回避とinput signatureの比較根拠。本成果物にTensorFlow実装はない|
|D6|[OpenCV公式DNN API](https://docs.opencv.org/doc/doxygen/html/de/db7/api_dnn_engine_selection.html)|モデル読込と実行エンジンの比較根拠。本成果物にOpenCV実装はない|

## GitHubの比較範囲

公式GitHub APIの確認値を[evidence/repository-evidence.json](../evidence/repository-evidence.json)に記録。2026-10-02 UTC時点ではESPnet 9,976、GTCRN 748、RealMAN 179 Stars。これはこの候補集合内の順位で、GitHub全体の上位順位ではありません。汎用ツールと個別モデルをStar数だけで順位づけて採用しません。

再現性について実際に確認するのは、コード・重み・前処理・固定版・動作試験の有無です。Starsは関心の代理指標で、学習データの独立性、論文スコア再現、商用利用可能性を証明しません。「高Starかつ再現性が高い」との第三者評価を確認できたとは主張しません。

`SamsungLabs/FINALLY`という推定パスは公式APIで404。論文は採用しますが、このパスを公式実装として案内せず、利用可能な実装の存在を断定しません。出典未確認の代替フォークにも置換していません。
