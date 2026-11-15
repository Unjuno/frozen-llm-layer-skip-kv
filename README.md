# Frozen LLM Layer Skipping × Deferred KV Reconstruction

> **Research design repository / 研究方法書（計画段階）**  
> This public repository specifies *how to investigate* dynamic layer skipping and deferred KV-cache reconstruction. **No experimental result, speedup, accuracy improvement, or novelty claim has been validated here yet.**

## 研究目的

**凍結した事前学習済みLLMの実行経路と、スキップ層のKVキャッシュ補完を共同最適化し、回答品質を維持または改善しながら、実GPU推論時間を短縮できるかを検証する。**

この研究では、**モデル本体の全重みを変更せず、Linear Routerのみを学習**する。最初と最後のTransformer層は常に実行する。第1層の出力から、**中間層すべてに対するON/OFFマスクをトークンごとに一括予測**する。各層を独立に評価するのではなく、選択した**マスク系列全体を実際の生成結果によって評価**し、Routerを更新する。

元モデルの出力を忠実に模倣することだけを目的にしない。層のスキップによって元モデルの誤答が正解になる可能性も、独立した研究仮説として検証する。

## 対象モデル・固定仕様

| 項目 | Phase 1: 手法確立 | Phase 2: 大規模検証 |
|---|---|---|
| モデル | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | [Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B) |
| Transformer層 | 28 | 52 |
| 必須実行層 | 第1・28層 | 第1・52層 |
| ON/OFF候補 | 第2〜27層（26ビット） | 第2〜51層（50ビット） |
| 学習対象 | Linear Routerのみ | 新規Linear Routerのみ |
| 転用するもの | 手法・評価プロトコル | Router重みは再学習 |

- Embedding、Transformer全層、LM Headの重みは**すべて凍結**。
- Linear RouterはTransformer層間の直列変換器ではなく、**中間層全体の実行マスクを一括出力する制御器**。
- 初期条件ではPrefillを全層実行し、Decodeのみ動的Layer Skipping。
- 3B・7B等の中間モデルは用いない。
- **週単位の期限や締切は設けず**、検証ゲートを満たすごとに次へ進む。

## 研究手法の要約

1. **正確性の基盤** — 元モデルと全ONマスク実行器を比較し、任意の中間層マスクとEager KVの正しさを確認。
2. **生成中の探索・学習** — Routerが出す全中間層の実行確率から、探索ノイズを含めてマスクをサンプリング。複数プロンプト×複数Rolloutで生成し、タスク品質と実行費用からRouterのみを学習。最初の候補はLeave-one-out基準化付きREINFORCE。
3. **品質維持・改善の検証** — 元モデルとの一致だけでなく、未見問題での正答率、改善件数・悪化件数を測定。
4. **KVの遅延補完** — Eager、On-demand、Batched、連続スキップ区間の入力状態共有、Adaptive Eager/Lazyを比較。同一マスク・同一トークン系列でKV方式だけを切り分ける。
5. **共同最適化** — 補完待ち量、GPU同期、メモリ、実測から較正した費用をRouterの学習に反映。Schedulerを後付けした方式との追加利益を評価。
6. **33Bで再検証** — Nanbeige用のRouterを新たに学習し、品質・GPU実時間・VRAMを検証。

## 反証可能な仮説

| 仮説 | 検証対象 |
|---|---|
| H1 | 学習済みLinear Routerは、同程度の平均実行層数の固定スキップより良い品質・費用トレードオフを示すか |
| H2 | 同じ実行マスク・生成トークン列で、遅延・一括・適応的KV補完がEagerより低遅延または省メモリになるか |
| H3 | **共同最適化**はRouter学習後にSchedulerを後付けした単純結合より有効か |
| H4 | 層スキップで元モデルを上回る正解率が、**未知プロンプト**でも再現できるか |

**最終的な主指標は品質制約付きの実測GPU Decode時間**。層スキップ率・FLOPsは補助指標。失敗例・否定的結果も記録対象とする。

## ドキュメント

- [研究実験方法の詳細](docs/RESEARCH_PROTOCOL.md) — 数理仕様、学習方式、KV補完の不変条件、対照実験、GPU評価、H/T/D/C/U、検証ゲート。
- [リポジトリ運用規約](docs/REPOSITORY_POLICY.md) — \`main\`での直接作業、変更の記録、再現性、公開時の注意。

現段階は**ドキュメントのみ**。実装用の\`src/\`・\`tests/\`・\`configs/\`・\`results/\`は必要になってから追加する。実装済み・測定済みと誤認させる空の雛形は作らない。

## ライセンス

[MIT License](LICENSE)。このリポジトリのMITライセンスは、第三者のモデル重み・データセット・外部コードに対する権利を与えるものではない。
