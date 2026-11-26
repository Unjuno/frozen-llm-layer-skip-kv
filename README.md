# Frozen LLM Layer Skipping × Deferred KV Reconstruction

> **状態：実験起動可能。小型Qwen3 CPU統合テスト13件PASS、事前学習済みQwen3-0.6BのFP32 CPU初回数値検証PASS（6マスク列・2 Decode位置）。** [小型実測](results/qwen3_synthetic_cpu_2026-11-25.json)／[実モデル実測](results/qwen3_real_cpu_fp32_2026-11-25.json)。実モデルのHead学習Pilotは実施済みだが品質改善・GPU速度は未検証。期限なし。

## 何を研究するか

**凍結したLLMの第1層Self-Attentionで既に計算した活性を再利用し、単一のLinear Headが、トークンごとに「どの中間層を実行するか」を一括学習する。** 生成結果の品質とGPU実費用で実行マスクを評価し、スキップ層で欠けるKVを必要時に正確に補完する。

この研究では、**まず動的な実行領域の学習が主題**。KV補完負債・将来需要・GPUスケジューリングは、実行領域選択の効果を正しく成立・評価させるための**従属課題**とする。

### 変えない原則

1. **第1層Attentionの再利用：** LLMの通常計算から得た活性をLinear Headも参照する。Head専用の第2のAttentionや別のLLMは実行しない。元の第1層は**MLPまで完成**し、その出力を中間層に渡す。
2. **唯一の学習対象：** Linear Headの重み・biasのみ。Embedding、すべてのTransformer層、LM Headは凍結する。
3. **トークンごとの全中間層マスク：** 第1層と最終層は必ず実行。間の層すべてについて、各DecodeトークンのON/OFFをHeadが**一括**出力する。
4. **生成中の学習：** ランダムに選択したマスク経路で実際に文章生成し、**軌跡全体の品質と計算費用**でHeadを学習する。元モデルの誤答が改善される可能性も検証する。
5. **KV補完：** 後から実行される層が参照する過去KVを、当該経路の正しい層入力から準備する。SchedulerはHeadのマスクを無断変更しない。
6. **固定層Skipへの誘導なし：** 同じマスクの維持や末尾OFF長を学習報酬にしない。固定マスクは比較対象である。

## 対象モデル

| 第1段階 | 第2段階 |
|---|---|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | [Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B) |
| 28層：中間26層を一括制御 | 52層：中間50層を一括制御 |
| 基本方式・正確性と学習方法を研究 | Routerを**新たに学習**して大規模側で検証 |

Prefillは当初全層実行し、Decode時だけ動的に制御する。3B・7B等の中間モデルは挟まない。

## 実験を始める

**最短手順（ネットワーク不要の小型モデル試験は依存導入後）：**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python scripts/run_pilot.py --stage smoke
```

実モデルで正確性を検証する場合：

```bash
python scripts/run_pilot.py --stage real --device cuda
```

検証→学習→開発評価→参照速度計測をまとめて実行する場合：

```bash
python scripts/run_pilot.py --stage pilot --device cuda
```

各実行は新規の `runs/` 配下にログ、JSON、manifest、必要な場合だけHeadのcheckpointを保存する。**エラーまたは学習更新ゼロで次工程を停止**する。CPUのみなら `--device cpu` を明示する。GPUリソースを自動契約する機能はない。詳細は[Runbook](docs/13_EXPERIMENT_RUNBOOK.md)。

**個別のコマンドやCPUでの再現条件は[13 Qwen3 Experiment Runbook](docs/13_EXPERIMENT_RUNBOOK.md)を参照。**

実モデルは必ずCheckpointの**完全なRevision SHA**を指定します。第1Attentionを共有する参照実行器：[qwen3_reference.py](src/frozen_skip/qwen3_reference.py)、検査・学習CLI：[run.py](experiments/qwen3/run.py)、一括KV射影のShape別ベンチ：[bench_kv_projection.py](experiments/qwen3/bench_kv_projection.py)。

**小型Qwen3と実モデルの短いFP32 CPU経路の一致を確認済み。長系列・BF16・GPU上の一般的な動作保証ではありません。** 学習後の開発データ評価には[evaluate.py](experiments/qwen3/evaluate.py)を使用します。GPU速度・品質改善は実証していません。GitHub Actionsの[CPU合成Qwen3検証ワークフロー](.github/workflows/qwen3-synthetic-cpu.yml)は手動起動専用であり、成功実績を意味しません。

## 研究文書を読む順番

| 優先 | 文書 | 役割 |
|---|---|---|
| **1** | [01 研究の骨格](docs/01_FOUNDATIONS.md) | **目的・確定仕様・行わないこと** |
| **2** | [11 活性共有とLinear Head](docs/11_SHARED_ACTIVATION_ROUTING.md) | **第1Attentionの共有位置とLLM/Headの依存関係** |
| **3** | [02 Linear Headの学習](docs/02_ROUTER_TRAINING.md) | 生成経路を使った学習の基本原理 |
| 4 | [03 正確なKV補完](docs/03_KV_CACHE.md) | Skipによって欠けるKVの保存・補完 |
| 5 | [07 学習方策の候補](docs/07_POLICY_LEARNING_PIPELINE.md) | REINFORCE/RLOOと任意のGPU費用反映 |
| 6 | [04 比較と評価の設計](docs/04_EVALUATION.md) | 動的経路と固定Skipの公平な比較 |

**補助資料：** [文書ガイド](docs/README.md)／[設計判断ログ](docs/DECISIONS.md)／[先行研究・出典台帳](docs/PRIOR_ART_AND_ATTRIBUTION.md)。KV負債、初回需要、最終ON位置等の研究候補は、ガイドに分類して保存するが**主設計を置き換えない**。

## 現在の状態

- **同一プロンプト内Pilot（24問×6位置）：** 事後OracleのKLはプロンプト固定0.05935→トークン別0.03572。全12評価プロンプトで事後最良マスクが変化。一方、初段Attention線形はEmbedding線形と全72位置で同じ選択、プロンプト単位線形より悪く、主要合否条件未達。**トークン別選択の改善余地は確認、初段活性の追加利益は未実証**。[Pilot Dの結果](docs/14_EARLY_ACTIVATION_PILOT.md)／[全144位置の実測](results/qwen3_within_prompt_cpu_2026-11-26.json)。


- **再検証（60問、12候補）：** 条件付きNLLは固定1.0075→初段Attention 0.9283だが、95%対応付き区間はゼロを含み、正解数14→13。選択は問題種別だけで説明でき、Native成立性も未達（12/24）。動的優位は未実証。[Pilot C](docs/14_EARLY_ACTIVATION_PILOT.md#pilot-c-60問12固定候補独立調整による再検証)／[全実測](results/qwen3_early_activation_v2_cpu_2026-11-26.json)。


- **2026-11-26 初段Attentionの新規性Pilot：** 実Qwen3のHeadを3回更新したが、未学習2問で全ON、Nativeを含め全方式0/2。別の25層固定予算診断（学習6問・比較4問）でもAttention線形予測は学習側最良固定を上回らず。**動的選択の優位性は未実証**。[結果・先行研究・次の合否条件](docs/14_EARLY_ACTIVATION_PILOT.md)。


- **一括実行の起動確認：** `python scripts/run_pilot.py --stage smoke --device cpu` の全4工程が終了コード0（[実行manifest](results/qwen3_one_command_smoke_2026-11-25.json)）。CPU環境の実測依存バージョンは[制約ファイル](requirements/cpu-tested.txt)。GPU環境は別途構築する。


- **Qwen3構造のCPU統合試験：** 公式クラスをランダム初期化した4層・隠れ幅64のモデルで13テストPASS。Tap両候補の全ON・Eager/Lazy比較がPASS（[結果](results/qwen3_synthetic_cpu_2026-11-25.json)）。
- **数学的な仮説：** 未確定と明示して分離。
- **CPUの合成モデル参照実験：** [初回正確性結果](docs/12_CPU_CORRECTNESS_PILOT.md)を取得済み。**Qwen3実モデルの結果ではない**。29種類のマスクでEager/LazyのLogits差0、位置誤りの負の対照も検出。
- **実Qwen3-0.6B初回検証：** CPU FP32、1プロンプト、2 Decode位置、6固定マスク列で全ONとNativeの差0、Eager/Lazyの最大Logits差0（[測定結果](results/qwen3_real_cpu_fp32_2026-11-25.json)）。
- **Qwen3のHead学習は初回Pilot済み、Nanbeige学習とGPU測定は未実施**。実測した高速化・品質改善・学術的新規性は主張しない。
- **活性を読み取る厳密なTensor位置：** Attention出力射影の前後、残差加算の前後などを混同しない。[11](docs/11_SHARED_ACTIVATION_ROUTING.md)で**未決**として管理する。

[MIT License](LICENSE)はこのリポジトリ独自の資料に適用する。モデル重みや外部資産のライセンスは別である。
