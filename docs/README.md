# 研究文書ガイド

> 研究設計に加え、**CPU合成モデルの参照正確性実験**を実施済み。**Qwen3本体とGPUの実験は未実施**。研究の正本は01と11。KV負債や最終ON位置は従属課題。

## まず読む文書：研究の中心

| 順番 | 資料 | 何を決めるか |
|---|---|---|
| **1** | [01 研究の骨格](01_FOUNDATIONS.md) | 凍結する範囲、目的、全中間層の実行領域、変えない前提 |
| **2** | [11 第1Attentionの活性共有とLinear Head](11_SHARED_ACTIVATION_ROUTING.md) | **どのAttention活性を共有するか、Headの役割、LLM本体との分岐** |
| **3** | [02 Linear Headの学習](02_ROUTER_TRAINING.md) | **生成中に全中間層マスクを選択し、品質・費用でHeadだけ更新する原理** |
| 4 | [03 正確なKV Cache](03_KV_CACHE.md) | 動的経路に応じた欠損KVの補完 |
| 5 | [07 方策学習の候補](07_POLICY_LEARNING_PIPELINE.md) | REINFORCE、RLOO、後段のGPU費用認識候補 |
| 6 | [04 比較方針](04_EVALUATION.md) | 動的Router、固定Skip、KV方式を区別する評価設計 |

## Qwen3で新たに実験するためのコード

- **[13 Qwen3実験Runbook](13_EXPERIMENT_RUNBOOK.md)**：固定モデルRevision、合成Qwen3検査、実モデルの数値確認、RLOO Pilot、GPU壁時計測定、正方行列理論と実KV GEMMの区別。
- [Qwen3参照実行器](../src/frozen_skip/qwen3_reference.py)／[verify・bench・train CLI](../experiments/qwen3/run.py)／[KV射影Microbenchmark](../experiments/qwen3/bench_kv_projection.py)。**新コードは未統合実行**。
- [小型Qwen3構造の検査](../tests/test_qwen3_reference.py)はモデル重みのダウンロード不要。実行後に初めて結果を記録する。

## 実際に実施した参照実験

- [12 CPU正確性Pilot](12_CPU_CORRECTNESS_PILOT.md)：**実行済み**。小型合成Decoderで共有Tap、全ON、KV補完、Headだけの勾配を確認。Qwen3の正確性やGPU速度を示すものではない。
- [CPU実行コード](../experiments/cpu_reference/run_cpu_reference.py)／[CPUの生結果JSON](../results/cpu_reference_2026-11-18.json)を公開。

## 補助資料・新規性の検討

| 文書 | 位置付け |
|---|---|
| [06 KV/GPU費用の最適化余地](06_OPTIMIZATION_TARGETS.md) | **主題を支える費用分析** |
| [08 KVの初回需要](08_KV_FIRST_USE_MODEL.md) | 将来の需要確率と補完費用に関する**理論的補助分析** |
| [09 最終ONとKVの必要数](09_LAST_USE_BOUND.md) | 生成後の条件付き計数命題。**同じ層のSkipを長くする学習目的ではない** |
| [10 新規性・合理性の監査](10_NOVELTY_AND_RATIONALITY.md) | 既知技術との境界、未検証の独自性候補 |
| [先行研究・出典台帳](PRIOR_ART_AND_ATTRIBUTION.md) | 参照文献、使った着想、外部資産の権利 |
| [設計判断ログ](DECISIONS.md) | 方針の訂正・撤回・未確定事項 |
| [05 将来の検証ゲート](05_GATES.md) | **実装依頼があるまでは着手しない**、期限のない将来の検証条件 |
| [リポジトリ運用規約](REPOSITORY_POLICY.md) | Public・`main`への直接コミット・公開ルール |

旧URLからの入口として[RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md)も残す。

## 事実と提案を混同しない

- **確定：** 第1Attentionの計算済み活性をLLMとLinear Headで共有。LLM重み全凍結。Headだけ学習。第1・最終層は必須。全中間層のON/OFFをトークンごとに一括選択。
- **未決：** Headが読むAttention活性の**残差加算前／後**の具体的なTap位置。学習・GPU Backendの細部。
- **提案：** REINFORCE/RLOO、KV Scheduler、補完費用による追加学習。
- **撤回：** 同じマスクを維持させる報酬やバイアス。
- **未検証：** 品質改善・GPU高速化・学術的な新規性。

**変更の原則：主目的を勝手にKV Cacheや静的Layer Pruningへ置き換えない。**
