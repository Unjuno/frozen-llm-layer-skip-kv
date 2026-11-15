# 先行研究・技術的出典台帳（Prior Art & Attribution）

> **2026-11-15時点の設計参照。** 本書は「何をどの研究から参考にしているか」と「本研究の未確定の差分」を追跡するための文書。**研究コード・図表・データセットの流用は、現時点で一切していない。** 実験上の改善・独自性もまだ実証していない。

## 1. この記録の意味

先行研究は積極的に参考にする。ただし、次を別の事実として管理する。

| 状態 | この台帳での意味 |
|---|---|
| **原理参照** | 数理原理・実装上の考え方を読んで研究設計の根拠にしている。手法の実装・コードのコピーを意味しない |
| **設計候補に反映** | 公開済みの研究構想に類似した仕組みを検討中。利用・実装・有効性確認は未実施 |
| **比較候補** | 独自性・実験設計上の重要な対照研究だが、その手法を採用したとは限らない |
| **外部資産の流用** | コード・モデル重み・データ・図・文章などを実際にコピー/改変・同梱した場合だけ使う。現在**該当なし** |

**原則：** 「文献を読んだ／参照した」≠「論文のアルゴリズムを実装した」≠「外部コードを複製した」。これらは独立して記録する。

## 2. 出典と本研究の対応（一次資料優先）

| ID | 研究（タイトル・原典） | 何に参考にしたか | この研究との差分・注意点 | 状態 |
|---|---|---|---|---|
| **RW-01** | Williams (1992), [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://doi.org/10.1007/BF00992696), DOI: **10.1007/BF00992696** | 確率的な二値マスク選択に対するREINFORCE型の方策勾配。[`02_ROUTER_TRAINING.md`](02_ROUTER_TRAINING.md)の導出の基礎 | **REINFORCE自体は本研究の独自性ではない**。Leave-one-out基準化などは学習の実装候補であり未実証 | 原理参照・設計候補に反映 |
| **RW-02** | He et al. (2025), [Router-Tuning: A Simple and Effective Approach for Dynamic Depth](https://aclanthology.org/2025.emnlp-main.99/), DOI: **10.18653/v1/2025.emnlp-main.99** | 凍結したBackboneに対してRouterを学習する先行例 | 「Backboneを凍結してRouterだけ学習」は**既知の考え方**。本研究は第1層特徴から全中間層マスクを一括予測し、生成報酬とKV負債を検討 | 比較候補・原理参照 |
| **RW-03** | Luo et al. (2025), [Adaptive Layer-skipping in Pre-trained LLMs (FlexiDepth)](https://arxiv.org/abs/2503.23798), arXiv:**2503.23798**（COLM 2025） | トークン単位の動的Layer Skippingと、凍結済みモデルへの後付け制御の先行例 | FlexiDepthはRouterとAdapterを使う。本研究は**Linear Routerのみ学習**し、初期段階ではAdapterを置かない。動的Skipが実GPUで速いとは限らない | 比較候補・原理参照 |
| **RW-04** | Heakl et al. (2025), [Dr.LLM: Dynamic Layer Routing in LLMs](https://arxiv.org/abs/2510.12773), arXiv:**2510.12773** | 学習で選んだ層経路が元モデルの品質を**改善し得る**こと、および凍結LLMの経路選択の先行例 | Dr.LLMはMCTSによる教師探索と層別Router等を扱う。本研究の中心は**生成中の確率的な全中間層マスク学習**とKV補完費用。品質改善自体を新規性と主張しない | 比較候補・仮説の先行根拠 |
| **RW-05** | Fu et al. (2024), [LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM Inference](https://arxiv.org/abs/2407.14057), arXiv:**2407.14057** | 不要な計算を遅延し、後に必要なコンテキスト情報のKVを生成するという研究上の関連例 | 主対象は**トークンのPruning**。本研究の主対象は**Transformer層のSkip**であり、同一手法ではない。単なる「遅延KV」の新規性は主張しない | 比較候補 |
| **RW-06** | Qasim et al. (2026), [The Residual Stream Is All You Need: On the Redundancy of the KV Cache in Transformer Inference](https://arxiv.org/abs/2603.19664), arXiv:**2603.19664** | 層入力の残差状態が与えられればK/Vを再構成できるという先行研究の理論的・技術的背景 | 本研究では**実際に選ばれたSkip経路の層入力・位置**から欠損KVを作る。同一マスクでの同値性とGPU時間は本研究側で別途検証が必要 | 原理参照・設計候補に反映 |
| **RW-07** | Li et al. (2026), [AnchorKV: Lossless Key-Value Cache Reconstruction from Layer-Sparse Residual Anchors](https://arxiv.org/abs/2611.00117), arXiv:**2611.00117** | GQAにおける残差保存とKV保存の損益分岐、アンカー・再計算のトレードオフ | **残差状態の共有・再構成や容量損益分岐は既知**。本研究の差分候補はトークンごとの**Layer Skip経路**と**動的な未補完KV負債**に応じて判断する点 | 原理参照・比較候補 |
| **RW-08** | Da et al. (2026), [vSkipper: Translating Dynamic Layer Skipping into LLM Serving Gains](https://arxiv.org/abs/2609.37062), arXiv:**2609.37062** | Layer SkippingがGPUの実時間高速化に直結しない問題、実行利益の判定とServing最適化 | GPUでSkipを速くすること自体は既知。本研究は**KV補完負債を含めた経路学習**の追加的な利益を検討する | 原理参照・比較候補 |
| **RW-09** | Ainslie et al. (2023), [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://aclanthology.org/2023.emnlp-main.298/), DOI: **10.18653/v1/2023.emnlp-main.298** | KV Head数と隠れ幅の関係、GQA構造がKVと保存入力の容量比較に与える影響 | GQA自体の設計を変更しない。対象モデルの既存GQA設定に依存する容量計算の背景文献 | 原理参照 |
| **RW-10** | Raposo et al. (2024), [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258), arXiv:**2404.02258** | トークン単位に深さ方向の計算資源を配分する研究の基本概念 | 既存MoDの学習方法・top-k予算を流用したとは扱わない。本研究ではトークンごとの**全中間層二値マスク**と、固定重み・生成報酬を扱う | 関連背景・比較候補 |

### 同名論文を混同しない

**AnchorKV**という略称には、少なくとも異なる研究が存在する。

- **RW-07（本研究と関連）:** *Lossless Key-Value Cache Reconstruction from Layer-Sparse Residual Anchors*, [arXiv:2611.00117](https://arxiv.org/abs/2611.00117)。
- **別研究:** *AnchorKV: Training-Free KV Cache Compression via Joint Coverage Diversity and Semantic Importance Scoring*, [EMNLP 2026](https://aclanthology.org/2026.emnlp-main.2428/) — 主にキャッシュ保持対象の選択と圧縮の研究。

**今後、略称だけではなく「正式名称＋DOIまたはarXiv ID」で特定する。** 査読済み/プレプリントの状態も混同しない。

## 3. 概念ごとの「どこから・何を」

| 本研究の要素 | 参照元 | 本研究で独自性として**主張しないこと** | 未確定の差分候補 |
|---|---|---|---|
| 凍結Backbone＋Router後付け | RW-02, RW-03, RW-04 | Routerだけ学習すること | 第1層から全中間層ON/OFFを一括予測し、生成報酬で選択する全経路の最適化 |
| 確率的探索・報酬での学習 | RW-01 | REINFORCEそのもの | Layer Mask系列とKV補完負債を含む費用の扱い |
| 動的Layer Skip | RW-03, RW-04, RW-10 | トークンごとの深さ制御 | KVの需要と相互作用する動的経路を選ぶこと |
| KV再構成・保存方式 | RW-05, RW-06, RW-07 | 残差からKVを作ること、Eager/Lazyそのもの、容量損益分岐 | **Skip経路により発生する欠損KV**の必要性・期限・メモリ制約を経路評価へ反映 |
| 実GPU実行・費用判定 | RW-08 | 動的SkipをServingで高速化すること | Router再学習で**KV補完負債を考慮する効果**を単純結合と比較 |
| GQAのメモリ背景 | RW-09 | GQAの発明、KV Head共有 | 対象モデルの形状を所与としたスキップ補完の判断 |

**注意：** これらの差分が最終的に新規性を持つかは、さらに関連研究を調査して確定する。**組み合わせたことだけで独自性を証明できるわけではない**。

## 4. 将来、手法・コード・データを取り入れるときの記録方法

論文を参考にして**設計案を考えること**は、外部コード・データの利用許諾を得たことを意味しない。GitHubのMITライセンスは第三者資産には適用されない。

**各引用・採用項目について最低限記録する内容：**

- **出典**：RW番号、著者、正式題名、年、DOI/arXiv/公式ページ、参照した版。
- **参照した内容**：概念、式、アルゴリズム、実装方針、比較条件のどれか。
- **採用状況**：「比較のための参照」「設計案に反映」「方式を採用予定」「実装済み」「外部コード改変／同梱済み」を区別。
- **変更した点**：元の方法と何が違うのか、どこに依存するのか。
- **採用理由・限界**：どの仮説に関係するか、未検証の仮定。
- **外部資産の再利用時のみ**：原著のリポジトリ、**固定Commit SHA**、対象ファイル、ライセンス・著作権表示、NOTICE義務、改変箇所、再配布の可否。

### 追記テンプレート

```markdown
### RW-XX: [正式タイトル]
- 著者・年：
- 一次資料：DOI / arXiv / 学会公式ページ（対象バージョン）：
- 本研究で参照する部分：
- 状態：原理参照 / 比較候補 / 設計候補に反映 / 実装済み / 外部資産再利用
- 採用した要素と、本研究で変更した要素：
- どの設計文書に反映したか：
- 新規性を主張しない既知要素：
- 残る疑問・先行研究との重複の可能性：
- （外部資産の再利用時）コードのURL・Commit SHA・ライセンス・NOTICE：
```

## 5. 現状の確認

- **使用しているのは研究概念と比較枠組みの参照**。研究文書に反映した考え方は上表で明示。
- **外部の研究コード、モデル重み、データ、画像、長い引用文をこのリポジトリへ複製していない**。
- どのRouter方式・KV方式が実際に有効か、GPUで高速になるかは**未実装・未検証**。
- 先行論文で測定された性能を、こちらの実測結果として転記しない。
- この台帳は文献調査の完成宣言ではなく、**必要な先行研究と採用履歴を随時整理する正本**。

[文書ガイド](README.md) ／ [01 研究の骨格](01_FOUNDATIONS.md) ／ [02 Router学習](02_ROUTER_TRAINING.md) ／ [03 KV補完](03_KV_CACHE.md) ／ [04 比較実験](04_EVALUATION.md)
