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

## 2.1 今回の最適化設計で追加参照する研究

| ID | 一次文献 | 今回、具体的に参考にした要素 | 本研究での扱い |
|---|---|---|---|
| **RW-11** | Ahmadian et al. (2024), [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740), arXiv:**2402.14740** | REINFORCE型の単純な最適化と、同一プロンプト複数出力のLeave-one-out型報酬基準化（RLOO）を学習器の比較候補にする | **既知の分散低減設計**。本研究の新規性とは主張しない。Router学習専用に適用できるかは未検証 |
| **RW-12** | Achiam et al. (2017), [Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a.html), ICML 2017 / [arXiv:1705.10528](https://arxiv.org/abs/1705.10528) | 品質下限とGPU時間・メモリの**制約付き目的**を区別するための既存理論 | この論文のCPOアルゴリズムを実装・採用したわけではない。**報酬重みだけで制約を保証できない**という設計上の注意の参照 |
| **RW-13** | Ng, Harada & Russell (1999), [Policy invariance under reward transformations: Theory and application to reward shaping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf), ICML 1999 | KV補完負債の中間的な学習信号に対し、**ポテンシャル差**を使うと終端条件下で方策の順位を保てるという既知原理 | 未確定の追加学習方式。終端ポテンシャルが0などの条件が必要。**学習済み効果・新規性を主張しない** |
| **RW-14** | Shi et al. (2026), [Multi-Segment Attention: Enabling Efficient KV-Cache Management for Faster Large Language Model Serving](https://arxiv.org/abs/2606.02964), arXiv:**2606.02964**（システム名 *AsymCache*） | 再計算費用、KVキャッシュの利用、GPUカーネル効率を組み合わせる**費用考慮型Scheduler**の先行例 | 単純に「KV費用を考慮したJoint Scheduler」を新規性とはしない。対象はKVのキャッシュ管理であり、本研究は**Dynamic Layer Skipの学習方策が負債を発生させること**に注目 |
| **RW-15** | Laitenberger et al. (2026), [What Layers When: Learning to Skip Compute in LLMs with Residual Gates](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7e3491e922bfd199ea34ecafeb7380f0-Abstract-Conference.html), ICLR 2026 (*GateSkip*) | トークン単位の計算省略において、残差のGatingで経路と品質を調整する別方式 | GateSkipは微分可能な残差ゲートと計算予算を扱う。本研究は**凍結全層＋第1層一括Linear Router＋生成報酬＋欠損KV補完負債**。品質改善自体は既知の可能性 |

新規の出典は、[06 最適化対象](06_OPTIMIZATION_TARGETS.md)と[07 方策学習パイプライン](07_POLICY_LEARNING_PIPELINE.md)の**理論設計の参考**として記録する。これら論文のコード・データ・図表を複製した事実はない。

## 2.2 最終ON位置・正確性の境界に関係する追加文献

| ID | 一次文献 | 参照する点 | 今回の手法との差分・採用状況 |
|---|---|---|---|
| **RW-16** | Wang (2026), [RISK-KV: Quality-Aware Selective KV Cache Reconstruction for Distributed LLM Inference](https://arxiv.org/abs/2610.29544), arXiv:**2610.29544** | 欠損KV再構成の対象層・品質悪化リスクを考慮する近接研究 | **RISK-KVは異なる層からの回帰による近似再構成**を扱う。本研究の同一マスクでのKV補完は**保存した正確な層入力からの射影**が前提。アルゴリズム・コードの流用なし。**比較候補** |

### 本研究で導いた簡単な恒等式の扱い

[09 最終ON位置とKV射影](09_LAST_USE_BOUND.md)に記した「同一マスク列で最後に層をONにした位置より後のSkip位置は、この生成要求中にKVが一度も必要とならない」という関係は、**標準Causal Attention・継続なし・正確なProject-Only/Lazyの意味論**に基づく初等的な計数恒等式である。**数学的な新発見と主張しない**。

この恒等式の**研究上の利用**は、「Layer Routerの選択する全マスク系列が、結果的にどれだけのK/V射影を後で必要にするか」という内生的な費用構造を解析するためである。正しい数のKVが必要かという条件と、GPUのGEMM・同期・待機による実効時間は別の量なので、計数恒等式をGPU高速化の証拠として使わない。

[vSkipper（RW-08）](https://arxiv.org/abs/2609.37062)のProject-OnlyはSkip時にK/Vを射影し、Cacheの完全性を保つ参照方式。[AnchorKV（RW-07）](https://arxiv.org/abs/2611.00117)は残差AnchorからのKV再構成と保存再計算のバランス。[AsymCache（RW-14）](https://arxiv.org/abs/2606.02964)はKV Evictionと再計算のGPU費用を最適化する。**AsymCacheのPrefix再計算**と、**保存済み層入力からの直接K/V射影**の費用を同一視しない。

[10 新規性と合理性の監査](10_NOVELTY_AND_RATIONALITY.md)に、既知要素・未検証の共同最適化主張・反例をまとめる。

## 2.3 第1層Attention活性の共有：実装構造の参照

本研究の[11 共有活性とLinear Head](11_SHARED_ACTIVATION_ROUTING.md)は、**第1層Attentionを既存LLMとLinear Headで共通利用し、AttentionをRouter用に二重計算しない**という設計。これ自体は深層学習の一般的な共有表現構造であり、**新規性を主張しない**。

| ID | 一次の実装・モデル文書 | 確認した技術上の境界 | 本研究の採用状況 |
|---|---|---|---|
| **ARCH-01** | [Hugging Face Transformers: Qwen3DecoderLayer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) | 第1層Decoder BlockはSelf-Attention・残差加算の後にMLPを含む。**Attention出力とBlock完成出力は別** | 演算順序の参照。実装コードの複製なし |
| **ARCH-02** | [NVIDIA NeMo: Nanbeige4.3-33B](https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/nanbeige/Nanbeige4.3-33B) | モデルの宣言アーキテクチャは**LlamaForCausalLM**。第1層Attention/MLPの分割は実際のBackendとRevisionで照合する必要がある | モデル構造の参照。実装コードの複製なし |

**実装時に必要な注意：** Qwen3の`output_attentions`などで取得するAttention**重み**は、Linear Head用に共有する**Hidden Stateの活性**とは異なる。Attention出力の残差加算前後、MLPまで完了したBlock出力のどれをTapとするかを事前に指定する。

## 2.4 行列乗算指数の2026年11月24日プレプリント（参考のみ）

| ID | 一次資料 | 本研究で参照した内容 | **実装への採用状態** |
|---|---|---|---|
| **RW-17** | OpenAI, *The matrix multiplication exponent is nine fourths* (2026-11-24), [公式プレプリント](https://github.com/openai/math/blob/main/preprints/The-matrix-multiplication-exponent-is-nine-fourths-November-24-2026/README.md), [Lean形式化の範囲](https://github.com/openai/math/blob/main/lean/docs/379.md) | 正方行列乗算の漸近的な指数を`9/4`と主張する理論と、実装上の長方形K/V射影を**混同しないため**の比較背景 | **参考のみ。アルゴリズム・コードは採用していない。独立したGPU高速化の証拠ではない** |

**出典上の注意：** この公表はプレプリントであり、著者の主張と学界による独立検証の完了を区別する。LeanのScopeは平方指数の形式化と、各有限行列サイズに対する具体的な下界の**未形式化**を明示する。この理論から本研究のGPU上のK/V射影速度を保証しない。[13 Experiment Runbook](13_EXPERIMENT_RUNBOOK.md)で、`M×d`と`d×2d_KV`の実際の長方形GEMMを別に計測する。

## 2.5 初段活性による経路選択の新規性と反証（2026-11-26）

| ID | 一次資料 | 本研究で使う点・違い | 状態 |
|---|---|---|---|
| **RW-18** | Luo, Wang, Yan, [DiffSkip: Differential Layer Skipping in Large Language Models](https://aclanthology.org/2025.findings-acl.377/), Findings ACL 2025, DOI:10.18653/v1/2025.findings-acl.377 | 同一層のAttention変化量をFFNの実行判断に使う既存手法。Attention由来のRouting信号一般を新規とは主張しない。本研究は第1Attentionの既存活性で全中間Blockを一括選択 | 比較・反証候補 |
| **RW-19** | Li, Li, Zhou, [Skip a Layer or Loop It? Learning Program-of-Layers in LLMs](https://proceedings.mlr.press/v306/li26cq.html), ICML 2026, [arXiv:2606.06574](https://arxiv.org/abs/2606.06574) | 凍結モデルの入力別プログラム選択・品質改善は既知。オフラインMCTSと予測器を用いる点に対し、本研究は既存Attention活性と一つのLinear Head、トークン単位の生成報酬学習を対象 | 比較候補 |
| **RW-20** | Westerhoff et al., [Programs-of-Layers in LLMs through the Lens of Cortical Areas](https://arxiv.org/abs/2609.31360), arXiv:2609.31360 (2026) | 追試ではPoLarの単一経路Routerの利益を再現できず、少数の汎用経路でも多くの問題を解けると報告。固定Maskを弱い恣意的Baselineにしない理由 | 独立追試の反証材料 |
| **RW-21** | Lin, Wan, Wang, [Do Dynamic Routers Need Memory? HeRo: History-Aware Routing for Efficient LLM Inference](https://arxiv.org/abs/2609.08189), arXiv:2609.08189 (2026) | 層をまたぐRouter履歴を扱うFFN制御。第1Attentionだけで後続層の状態依存性を近似できるかは未証明 | 比較・限界の参考 |

**RW-04の追加確認：** Dr.LLMのTable 6ではLLaMA 3.2-3Bで各層直前の特徴がARC 74.5%、DART 38.6%、入力Embeddingを全層Routerへ与える方式が70.9%、30.0%、元モデルが73.5%、35.2%。これは**Embedding**との比較であり、第1Attention残差後の活性の実験ではない。従って初段Attentionの共有活性が十分かを独立に試す価値がある。

**検証設計：** まず固定経路を開発データで選び、同一の実行層数と独立評価で比較する。初段AttentionとEmbeddingの情報差を測る補助診断は、[probe_early_activation.py](../experiments/qwen3/probe_early_activation.py)に実装する。これは候補3マスクの1-step教師強制NLLの線形読み出しであり、本研究の26-bitオンポリシー生成学習の置換・学習成果の証明ではない。コード・データ・図表の外部流用なし。網羅的な新規性調査の完了は主張しない。

## 2.6 固定入力の線形制御器・プロンプト単位経路との比較（2026-11-26）

| ID | 確認した一次資料 | 本研究の反証・比較に使う点 |
|---|---|---|
| **RW-22** | Glavas et al., [Dynamic layer selection in decoder-only transformers](https://arxiv.org/abs/2410.20022), 2024, [§3.2 / Appendix A.4.3](https://arxiv.org/html/2410.20022v1) | OPT-1.3Bにおけるトークン別の線形層選択器では、隠れ状態入力が定数入力に有意に勝らなかったと報告。初段Attentionの共有活性を使う一括マスクの情報価値は、入力非依存の強い対照で反証可能にする必要がある。 |
| **RW-23** | Jain et al., [FiRST: Finetuning Router-Selective Transformers for Input-Adaptive Latency Reduction](https://aclanthology.org/2025.findings-emnlp.1197/), Findings EMNLP 2025, DOI:10.18653/v1/2025.findings-emnlp.1197 | Prefillのプロンプトに応じてDecodeの層を決める経路制御。入力適応そのものは新規性にできない。トークン別全層一括マスクの価値は、プロンプト単位の経路制御に対しても検証する。 |

いずれも本研究の同等手法・同等性能を証明するものではない。論文の存在と上記主張を一次資料で確認したが、網羅的調査は未完了。第1Attention活性の情報量については、固定マスク集合を増やし、独立した学習・調整・評価データを使う補助診断 [probe_early_activation_v2.py](../experiments/qwen3/probe_early_activation_v2.py) で調べる。新規性を証明したとは扱わない。

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

### 追加の引用の対応：初回KV需要と補完負債

[08 初回KV需要の数理モデル](08_KV_FIRST_USE_MODEL.md)は、条件付き初回需要の確率分解という**一般的な確率論上の恒等式**に、以下の既知システム技術の問題意識を組み合わせた**提案設計**である。

- **RW-08（vSkipper）**：SkipトークンでもKVを**即時に射影・保存**する`Project-Only`によりキャッシュの完全性を維持する。**提案するLazyはこの射影自体の遅延**を検討する点が異なる。
- **RW-07（Residual AnchorKV）**：保存状態からのKV再構成とメモリ・再計算の損益分岐。再構成や容量閾値を独自の発明とはしない。
- **RW-14（AsymCache）**：GPU実行費用に合わせたKVの管理・補完。GPUコストをSchedulerで考慮することだけを新規性とはしない。
- **RW-04（Dr.LLM）**：凍結モデルのLayer Routing。Router後付けや経路の品質改善自体は既知。

したがって、検討中の差分は**「トークンごとの全中間層マスクの選択が、将来の初回KV需要・補完バーストを作り出す」という因果関係をRouterの学習目的に取り込むこと**であり、既存研究に同等の手法がないことは未確認。新しいコードや外部資産の再利用はしていない。

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
