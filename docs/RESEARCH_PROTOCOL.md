# Research Protocol｜研究方法の入口

> **現在は設計のみ**。研究コード、学習・GPU実験、結果、期限は設定しない。研究目的と厳密な計算の共有範囲は[01](01_FOUNDATIONS.md)・[11](11_SHARED_ACTIVATION_ROUTING.md)が正本。

## 何をする研究か

既存LLMの**第1層Self-Attentionで計算済みの活性を再利用**し、**一つのLinear Head**がトークンごとに**中間層すべてのON/OFF（実行領域）**を一括選択する。LLMのEmbedding・全Transformer・LM Headは凍結し、Headだけを**実際の生成報酬**から学習する。第1層はMLPまで実行し、最終層も必ず実行する。

中間層のSkipによって欠けたKVは必要時に正しく補完する。このKV補完とGPU費用の最適化は、上記の**動的経路学習を支える派生課題**であり、主題を置き換えない。

## 目的別の参照先

| 対象 | 正本文書 |
|---|---|
| **研究の確定仕様と目的** | [01 研究の骨格](01_FOUNDATIONS.md) |
| **第1Attentionの活性再利用・Linear Headの位置** | [11 共有活性の仕様](11_SHARED_ACTIVATION_ROUTING.md) |
| **Linear Headの生成中学習** | [02 学習方法](02_ROUTER_TRAINING.md) |
| 中間層の欠損KVの正確な補完 | [03 KV Cache](03_KV_CACHE.md) |
| 学習方策・任意の費用拡張 | [07 学習候補](07_POLICY_LEARNING_PIPELINE.md) |
| 評価の対照設計 | [04 比較方針](04_EVALUATION.md) |
| **KV費用などの補助分析** | [06](06_OPTIMIZATION_TARGETS.md)・[08](08_KV_FIRST_USE_MODEL.md)・[09](09_LAST_USE_BOUND.md) |
| 既知技術との境界 | [10 新規性監査](10_NOVELTY_AND_RATIONALITY.md)・[出典台帳](PRIOR_ART_AND_ATTRIBUTION.md) |
| 将来の検証ゲート、決定履歴、運用 | [05](05_GATES.md)・[判断ログ](DECISIONS.md)・[運用](REPOSITORY_POLICY.md) |

## 運用上の注意

- **単一Linear Headが中間層全体のON/OFFを選ぶ。** 別のAttention、別の学習可能なDecision Modelを足さない。
- Attention重みとAttention出力活性を混同しない。厳密なTap位置は未決で、勝手に確定しない。
- **同じ層をSkipし続けることを学習目標にしない。** 固定SkipはBaseline。
- 数理解析、推論、実装、実測、学術的新規性の確定を混同しない。
- [文書ガイド](README.md)から現在の正本へ進む。
