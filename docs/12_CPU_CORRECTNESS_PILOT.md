# CPU参照実験 01：共有Attention活性・Layer Mask・遅延KVの正確性

**実行日：2026-11-18 (UTC)**  
**結果の種類：実行済みのCPU合成モデル実験**  
**範囲：小型合成Decoder。Qwen3-0.6Bの事前学習済み重みは使用していない。GPUは未使用。**

## なぜこの実験を先に行うか

研究の中心である「**LLM第1層Attentionの既存活性をLinear Headに再利用し、そのHeadだけでトークン別の全中間層実行マスクを選択する**」構造について、まず計算依存関係とKV補完の意味論に矛盾がないかを確認するため。

この実験は、LLMの品質改善・Qwen3との厳密一致・GPU高速化を検証するものではない。**それらをこの結果から主張してはいけない。**

## 実行環境・再現条件

| 項目 | 実際の条件 |
|---|---|
| CPU/GPU | CPUのみ（CUDAなし、1スレッド） |
| OS | Linux 6.18.57 x86_64, glibc 2.41 |
| Python / PyTorch | 3.13.5 / 2.10.0+cpu |
| 精度 | float64（BF16ではない） |
| モデル | **事前学習していない合成Decoder**：5層、隠れ幅16、Q heads 4、KV heads 2、head dim 4、語彙29 |
| 第1・最終層 | 常時実行。第1層のMLPまで完了 |
| 共有Tap | 第1層Attentionの出力射影と**残差加算後、MLP前**。**この実験限りの選択であり研究全体のTapを確定していない** |
| テスト系列 | 固定トークン8個、**29種類の中間3ビットマスク列**、8位置をReplay |
| 乱数Seed | Model 20261118、Masks 444、Linear Head 37 |
| KV補完 | Skipした位置の正しい層入力を保存。後続で層を実行する直前に、位置回転とKV射影を適用して補完 |
| 出力の同値性 | Logitsの要素最大絶対差、許容誤差1e-12 |

実装はRMSNorm、GQA形式のHead共有、簡略化した位置回転、残差・MLPを持つ**参照Decoder**。**Qwen3のRoPE/QK norm、実際のBackend、KV Layoutを再現したとの主張ではない。**

## 実測結果

| Test ID | 検査 | 実行した事実 | 判定 |
|---|---|---|---|
| CPU-01 | 通常全ON経路 vs 共有Tapを消費する全ONマスク経路 | Logits差**0.0**、Tap差**0.0** | **PASS** |
| CPU-02 | 共有による第1Attentionの二重実行がないか | 8トークンに対して第1Attention呼出し**8回**。Router側のTap読取り8回 | **PASS** |
| CPU-03 | 第1層MLPをスキップしていないか | Native全ON＋Masked全ONの計16トークン処理で第1層MLP呼出し**16回** | **PASS** |
| CPU-04 | EagerとLazyの固定トークン・マスクReplay | **29マスク列すべてLogitsの最大差0.0** | **PASS** |
| CPU-05 | 位置を誤って補完した場合の負の対照 | **Deferred KVの補完位置だけ**意図的に1ずらすとLogits最大差**0.021333182293028385** | **PASS：差を検出** |
| CPU-06 | REINFORCEでLinear Headのみ更新できるか | 6独立Rollout、Weight gradient norm **0.01702735675436272**、重み最大変化**0.00034620298151419315**。凍結モデルのSHA256は更新前後で一致 | **PASS** |

### KVの計算個数（固定Replayの代表例）

| ケース | Eagerが射影したKV組 | Lazyが射影したKV組 | 不要になった組 |
|---|---:|---:|---:|
| All-on | 40 | 40 | 0 |
| 中間3層すべてOff | 40 | 16 | 24 |
| 最初6位置Off、後の2位置On | 40 | 40 | 0 |
| 中間層交互On/Off | 40 | 39 | 1 |
| Mixed（Seed 444） | 40 | 37 | 3 |

ここで「組」とは**1層・1トークン位置についてのK/Vペア**。5層×8位置＝40組が全Eagerの参照で、これは計算個数でありGPU速度ではない。

**Eager/Lazyの数値一致は、同一トークン・同一マスク列という条件でのみ確認**。別の自己回帰生成経路での確率的一致や、実モデルでの全ON一致までは調べていない。

### Linear Headの勾配検査の解釈

6回の合成生成では、**正解タスクの成立件数は0件**。勾配が非ゼロだったのは探索経路の計算費用が異なったためであり、**回答品質を学習・改善できた証拠にはならない**。

したがって、今回のCPU-06から言えることは**凍結LLMに勾配を流さず、Linear Headだけを更新できる**という配線の成立までである。

## ERROR CHECK — 結果の適用範囲

- **PASSしたもの：** この小型Decoderにおける共有Tap、MLP常時実行、全ONの同値、Eager/Lazy再構成、位置ずれに対する感度、Linear Headだけの1ステップ更新。
- **未検証：** **Qwen3-0.6Bの実重み**、Qwen3固有のRMSNorm/QK norm/RoPE、実モデルの全ON Logits、BF16の丸め差、GPU上のGather/Scatter・CUDA同期・VRAM・Latency、未知問題の品質、Nanbeige4.3-33B。
- **重要な前提：** 合成Decoderは自作コード内部のNativeとMaskedを比較している。**Transformersの独立実装を比較してはいない**。
- **速度の結論：** GPUが利用できなかったため、**Speedupは「未測定」**。0%や高速化成功とするのは誤り。
- **実行順序上の限界：** CPU参照コードは第1BlockのMLPまで計算し、返されたAttention残差後TapをLinear Headが読む。**Headの判定と第1層MLPを並行実行した実証ではない**。共有値の正確性と、実GPUスケジューリング上の重なりは別課題。

## 保存先と再現手順

- CPU参照コード：`experiments/cpu_reference/run_cpu_reference.py`
- 実行結果：`results/cpu_reference_2026-11-18.json`
- コマンド：`python experiments/cpu_reference/run_cpu_reference.py --output results/cpu_reference_2026-11-18.json`

同等の結果を再現するには、上記Python・PyTorchのバージョン、CPUスレッド数、モデル初期化Seed、マスクSeed、Router Seedを揃える。異なるBackend・CPUでの絶対一致は一般に保証できない。

### 次の条件付き検証（まだ未実施）

Qwen3-0.6Bの実重み、Transformersの固定revision、対応するTokenizer/Chat Template、BF16/FP32と適切なGPU環境が利用可能になった段階で、**独立したNative Qwen3とMasked-All-Onの一致**および第1Attentionの正確なTap位置を確認する。

[研究の骨格](01_FOUNDATIONS.md)／[共有活性仕様](11_SHARED_ACTIVATION_ROUTING.md)／[比較設計](04_EVALUATION.md)
