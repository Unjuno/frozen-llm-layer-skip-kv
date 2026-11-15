# Frozen LLM Layer Skipping × Deferred KV Reconstruction

**凍結LLMの中間層をトークンごとに選んで実行し、欠損KVの補完と共同で最適化する研究。**

> **現在：研究方法の文書化段階**  
> モデルの実装・Router学習・GPU高速化・品質改善の実験結果は、まだ公開していません。**研究に期限は設けず、検証条件で次へ進みます。**

## 何を研究するか

- **重みは変更しない：** Embedding・全Transformer層・LM Headは凍結。学習対象は**Linear Routerのみ**。
- **中間層を選ぶ：** 第1・最終Transformer層は必ず実行。第1層後の特徴から**中間層すべてのON/OFFをトークンごとに一括選択**する。
- **生成しながら学習：** ランダムな実行マスクを使って実際に文章生成し、品質と推論費用からRouterを更新する。
- **元モデルを上回る可能性も評価：** 元モデルの模倣だけではなく、層スキップによる正解率改善も検証する。
- **KVを必要に応じて補完：** スキップした層の入力を保持し、Eager / On-demand / Batched / Shared Lazy / Adaptiveを比較する。
- **実GPU時間で判定：** 計算層数だけでなく、Router・同期・KV補完・VRAMを含む生成時間を測定する。

## 対象モデル

| 第1段階 | 第2段階 |
|---|---|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | [Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B) |
| 28層。第2〜27層がON/OFF対象 | 52層。第2〜51層がON/OFF対象 |
| 手法・数値整合性・GPU実測を検証 | 大規模モデルへの適用と性能を検証 |

**3B・7B等の中間モデルは使用しません。** Prefillは当初全層実行し、Decodeを動的に制御します。

## どこから読むか

**最初に読む： [研究文書ガイド](docs/README.md)**

| 文書 | 内容 |
|---|---|
| [01 研究の骨格](docs/01_FOUNDATIONS.md) | 目的・確定仕様・モデル構造 |
| [02 Routerの学習](docs/02_ROUTER_TRAINING.md) | 確率マスク・生成報酬・REINFORCE |
| [03 KV補完](docs/03_KV_CACHE.md) | 保存状態・補完方法・同一経路の正確性 |
| [04 比較実験と評価](docs/04_EVALUATION.md) | H1〜H4、アブレーション、GPU計測・統計 |
| [05 検証ゲート](docs/05_GATES.md) | **現在地・未決事項・次に進む条件** |
| [06 最適化対象](docs/06_OPTIMIZATION_TARGETS.md) | **改良余地の絞り込み・理論上の限界と対策（設計案）** |
| [07 学習パイプライン](docs/07_POLICY_LEARNING_PIPELINE.md) | **全マスク選択・報酬学習・KV負債を考慮する方策（設計案）** |
| [08 初回KV需要のモデル](docs/08_KV_FIRST_USE_MODEL.md) | **将来需要の確率・時刻・補完バースト・メモリの分離（設計案）** |
| [判断ログ](docs/DECISIONS.md) | 確定した方針と実装候補を区別して保存 |
| [先行研究・出典台帳](docs/PRIOR_ART_AND_ATTRIBUTION.md) | 何を参考にし、何を採用候補としたか／本研究との差分 |
| [運用規約](docs/REPOSITORY_POLICY.md) | `main`での直接更新・公開・記録ルール |

旧リンクからは[RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md)を入口として使えます。

## 研究の現在地

- **文書の整理：** 公開済み
- **Qwen3マスク実行器：** 未実装・未検証
- **Router学習／KV補完：** 未実装・未検証
- **GPU性能と品質改善：** 未測定
- **次に検証する段階：** [G1 — Qwen3の全層Baselineと任意マスクの正確性](docs/05_GATES.md)

検証が済むまでは成功を宣言せず、否定的結果も記録します。

## リポジトリの扱い

ドキュメントを研究の正本として、原則`main`へ直接コミットします。実装コード・設定・結果は、着手した時点で追加します。今は未作成の空ディレクトリは置きません。

このリポジトリのオリジナル資料は[MIT License](LICENSE)に従います。外部モデル・データセット・コードの利用条件は別途確認してください。
