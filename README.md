# Frozen LLM Layer Skipping × Deferred KV Reconstruction

> **状態：研究設計＋CPU小型合成モデルの初回正確性検査。** Qwen3本体による検証・GPU実測は未実施。期限なし。原則として公開リポジトリの`main`へ直接コミットする。

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

- **設計資料の整理：** 継続中。数学的な候補は「未確定」として分離。
- **実装・学習・GPU測定：** **未実施**。実測した高速化・品質改善・学術的新規性は主張しない。
- **活性を読み取る厳密なTensor位置：** Attention出力射影の前後、残差加算の前後などを混同しない。[11](docs/11_SHARED_ACTIVATION_ROUTING.md)で**未決**として管理する。

[MIT License](LICENSE)はこのリポジトリ独自の資料に適用する。モデル重みや外部資産のライセンスは別である。
